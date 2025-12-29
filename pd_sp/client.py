import socket
import torch
import threading
import time
from transformers import AutoTokenizer
from utils import *

import random
import queue

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class InferenceRequest:
    def __init__(self, req_id, prompt_len, output_len):
        self.req_id = req_id
        self.arrival_time = time.time() # 记录到达时间（进入队列的时间）
        self.prompt_len = prompt_len
        self.output_len = output_len
        # 模拟随机 Input Token (为了性能测试，直接生成随机Tensor比Tokenizer更快)
        # 词表大小假设为 32000 (Llama 标准)
        self.input_tokens = torch.randint(0, 32000, (1, prompt_len), dtype=torch.long)

class WorkloadGenerator(threading.Thread):
    def __init__(self, request_queue, arrival_rate, max_requests):
        super().__init__()
        self.request_queue = request_queue
        self.arrival_rate = arrival_rate # Lambda (requests/second)
        self.max_requests = max_requests
        self.stop_event = threading.Event()

    def run(self):
        print(f"[Generator] Started. Target Rate: {self.arrival_rate} req/s")
        for i in range(self.max_requests):
            if self.stop_event.is_set():
                break
            
            # 1. 生成请求
            req = InferenceRequest(req_id=i, prompt_len=4096, output_len=128)
            self.request_queue.put(req)
            print(f"[Generator] Request {i} arrived. Queue size: {self.request_queue.qsize()}")
            
            # 2. 泊松分布等待
            if self.arrival_rate > 0:
                sleep_time = random.expovariate(self.arrival_rate)
                time.sleep(sleep_time)
        
        print("[Generator] All requests generated.")

class EdgeClient:
    def __init__(self):
        print(f"[Edge] Initializing on {DEVICE}...")
        self.config = LlamaConfig()
        self.load_weights()
        self.init_rope()
        self.kv_cache = {} 
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        # [修改 1] 初始化 KV 发送缓冲区，用于 PD 分离模式下暂存 KV
        self.prefill_kv_buffer = {} 

        # 连接 Cloud
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("localhost", SERVER_PORT))
        self.send_lock = threading.Lock() 
        print(f"[Edge] Connected to Cloud Server.")

    def load_weights(self):
        # Edge 需要全量权重用于 Prefill 计算 (为了隐私，Edge 自己算所有层)
        full_weights = torch.load(
            os.path.join(MODEL_PATH, "original/consolidated.00.pth"), 
            map_location=DEVICE, weights_only=True
        )
        self.weights = {k: v.to(torch.bfloat16) for k, v in full_weights.items()}

    def init_rope(self):
        max_seq_len = 8192
        freqs = 1.0 / (self.config.rope_theta ** (torch.arange(0, self.config.head_dim, 2, device=DEVICE).float() / self.config.head_dim))
        t = torch.arange(max_seq_len, device=DEVICE, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    # [修改 2] 移除了 async_send_kv 线程函数，改为同步发送函数（或者直接在 send_buffered_kv 中处理）
    # 但为了兼容可能得直接调用，保留一个 send 逻辑，这里主要新增 send_buffered_kv

    def send_buffered_kv(self):
        """
        [修改 3] 新增函数：PD 分离核心逻辑
        在 Prefill 计算全部完成后，统一发送所有缓冲的 KV Cache 到云端。
        """
        if not self.prefill_kv_buffer:
            return

        print(f"[Edge] Transmitting buffered KV Caches ({len(self.prefill_kv_buffer)} layers)...")
        t_start = time.time()
        
        # 按层号顺序发送，虽然 Server 并不严格要求顺序，但顺序发送更可控
        sorted_layers = sorted(self.prefill_kv_buffer.keys())
        
        with self.send_lock:
            for layer_idx in sorted_layers:
                kv_data = self.prefill_kv_buffer[layer_idx]
                kv_bytes = serialize_tensor(kv_data)
                send_packet(self.sock, MSG_KV_CACHE, kv_bytes, layer_idx)
        
        # 发送完后清空缓冲区
        self.prefill_kv_buffer.clear()
        print(f"[Edge] KV Transmission done in {(time.time() - t_start)*1000:.2f} ms")

    def forward_layer_prefill(self, h, layer_idx, seq_len):
        """
        Prefill 层的计算
        """
        # 1. --- RMS Norm & QKV Projection ---
        norm_w = self.weights[f"layers.{layer_idx}.attention_norm.weight"]
        wq = self.weights[f"layers.{layer_idx}.attention.wq.weight"]
        wk = self.weights[f"layers.{layer_idx}.attention.wk.weight"]
        wv = self.weights[f"layers.{layer_idx}.attention.wv.weight"]
        wo = self.weights[f"layers.{layer_idx}.attention.wo.weight"]

        h_norm = rms_norm(h, norm_w, self.config.norm_eps)

        xq = torch.matmul(h_norm, wq.T).view(1, seq_len, self.config.n_heads, self.config.head_dim)
        xk = torch.matmul(h_norm, wk.T).view(1, seq_len, self.config.n_kv_heads, self.config.head_dim)
        xv = torch.matmul(h_norm, wv.T).view(1, seq_len, self.config.n_kv_heads, self.config.head_dim)

        # 2. --- RoPE ---
        freqs_cis_curr = self.freqs_cis[0 : seq_len].view(1, seq_len, 1, -1)
        xq = torch.view_as_real(torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) * freqs_cis_curr).flatten(3).to(torch.bfloat16)
        xk = torch.view_as_real(torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) * freqs_cis_curr).flatten(3).to(torch.bfloat16)

        # 3. --- KV Cache 处理 (修改：存入缓冲区而非异步发送) ---
        k_cache = xk.detach().clone().contiguous()
        v_cache = xv.detach().clone().contiguous()
        current_kv = {'k': k_cache, 'v': v_cache}

        # [修改 4] 仅将 Cloud 负责的层存入缓冲区，不立即发送
        if CLOUD_START_LAYER <= layer_idx < CLOUD_END_LAYER:
            self.prefill_kv_buffer[layer_idx] = current_kv
        else:
            self.kv_cache[layer_idx] = current_kv

        # 4. --- Attention Computation ---
        xq = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)

        if self.config.n_kv_heads < self.config.n_heads:
            n_rep = self.config.n_heads // self.config.n_kv_heads
            keys = keys.repeat_interleave(n_rep, dim=1)
            values = values.repeat_interleave(n_rep, dim=1)

        scores = torch.matmul(xq, keys.transpose(-2, -1)) / (self.config.head_dim ** 0.5)
        
        mask = torch.full((seq_len, seq_len), float("-inf"), device=DEVICE)
        mask = torch.triu(mask, diagonal=1)
        
        scores = scores + mask 
        probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        output = torch.matmul(probs, values)
        output = output.transpose(1, 2).contiguous().reshape(1, seq_len, self.config.dim)

        h_out = h + torch.matmul(output, wo.T)

        # 5. --- FFN ---
        norm_ffn_w = self.weights[f"layers.{layer_idx}.ffn_norm.weight"]
        h_norm_ffn = rms_norm(h_out, norm_ffn_w, self.config.norm_eps)
        
        w1 = self.weights[f"layers.{layer_idx}.feed_forward.w1.weight"]
        w2 = self.weights[f"layers.{layer_idx}.feed_forward.w2.weight"]
        w3 = self.weights[f"layers.{layer_idx}.feed_forward.w3.weight"]
        
        ffn_out = torch.matmul(
            torch.nn.functional.silu(torch.matmul(h_norm_ffn, w1.T)) * torch.matmul(h_norm_ffn, w3.T), 
            w2.T
        )

        return h_out + ffn_out

    def forward_layer_decode(self, h, layer_idx, start_pos):
        # Decode 逻辑保持不变
        seq_len = 1
        cache = self.kv_cache[layer_idx]
        
        norm_w = self.weights[f"layers.{layer_idx}.attention_norm.weight"]
        h_norm = rms_norm(h, norm_w, self.config.norm_eps)
        
        wq, wk, wv, wo = [self.weights[f"layers.{layer_idx}.attention.{name}.weight"] for name in ['wq', 'wk', 'wv', 'wo']]
        
        xq = torch.matmul(h_norm, wq.T).view(1, seq_len, self.config.n_heads, self.config.head_dim)
        xk = torch.matmul(h_norm, wk.T).view(1, seq_len, self.config.n_kv_heads, self.config.head_dim)
        xv = torch.matmul(h_norm, wv.T).view(1, seq_len, self.config.n_kv_heads, self.config.head_dim)
        
        freqs_cis_curr = self.freqs_cis[start_pos : start_pos + seq_len].view(1, seq_len, 1, -1)
        xq = torch.view_as_real(torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) * freqs_cis_curr).flatten(3).to(torch.bfloat16)
        xk = torch.view_as_real(torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) * freqs_cis_curr).flatten(3).to(torch.bfloat16)
        
        keys = torch.cat([cache['k'], xk], dim=1)
        values = torch.cat([cache['v'], xv], dim=1)
        self.kv_cache[layer_idx] = {'k': keys, 'v': values}
        
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        if self.config.n_kv_heads < self.config.n_heads:
             n_rep = self.config.n_heads // self.config.n_kv_heads
             keys = keys.repeat_interleave(n_rep, dim=1)
             values = values.repeat_interleave(n_rep, dim=1)
        
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / (self.config.head_dim ** 0.5)
        probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        output = torch.matmul(probs, values).transpose(1, 2).reshape(1, seq_len, self.config.dim)
        
        h_out = h + torch.matmul(output, wo.T)
        
        h_norm_ffn = rms_norm(h_out, self.weights[f"layers.{layer_idx}.ffn_norm.weight"], self.config.norm_eps)
        w1, w2, w3 = [self.weights[f"layers.{layer_idx}.feed_forward.{name}.weight"] for name in ['w1', 'w2', 'w3']]
        ffn_out = torch.matmul(torch.nn.functional.silu(torch.matmul(h_norm_ffn, w1.T)) * torch.matmul(h_norm_ffn, w3.T), w2.T)
        
        return h_out + ffn_out

    def run(self):
        # [修改 5] 同步更新 run 方法中的流程
        prompt = "The capital of France is"
        print(f"\nPrompt: {prompt}")
        tokens = torch.tensor(self.tokenizer.encode(prompt), device=DEVICE).unsqueeze(0)
        seq_len = tokens.shape[1]
        
        print("[Edge] Starting Prefill (PD Separation Mode)...")
        t0 = time.time()
        
        h = torch.nn.functional.embedding(tokens, self.weights["tok_embeddings.weight"]).to(torch.bfloat16)
        
        # Prefill 计算循环 (只计算 + 缓冲，不发送)
        for layer in range(self.config.n_layers):
            h = self.forward_layer_prefill(h, layer, seq_len)
        
        # [修改 6] Prefill 计算结束后，统一发送 KV Cache
        self.send_buffered_kv()

        h_final = rms_norm(h, self.weights["norm.weight"], self.config.norm_eps)
        logits = torch.matmul(h_final[:, -1, :], self.weights["output.weight"].T)
        next_token = torch.argmax(logits, dim=-1)
        time_to_first_token = time.time() - t0
        print(f"[Edge] Prefill Done. First token: {self.tokenizer.decode(next_token)}")
        
        print("[Edge] Starting Collaborative Decoding...")
        current_pos = seq_len
        
        token_gen_times = []
        for _ in range(128): 
            t_token_start = time.time()
            token_tensor = torch.tensor([[next_token]], device=DEVICE)
            h = torch.nn.functional.embedding(token_tensor, self.weights["tok_embeddings.weight"]).to(torch.bfloat16)
            
            for layer in range(0, CLOUD_START_LAYER):
                h = self.forward_layer_decode(h, layer, current_pos)

            req_payload = {'h': h, 'start_pos': current_pos} 
            req_bytes = serialize_tensor(req_payload)
            
            with self.send_lock:
                send_packet(self.sock, MSG_HIDDEN_REQ, req_bytes)
            
            _, _, res_bytes = recv_packet(self.sock)
            h = deserialize_tensor(res_bytes) 
            
            for layer in range(CLOUD_END_LAYER, self.config.n_layers):
                h = self.forward_layer_decode(h, layer, current_pos)
            
            h_final = rms_norm(h, self.weights["norm.weight"], self.config.norm_eps)
            logits = torch.matmul(h_final[:, -1, :], self.weights["output.weight"].T)
            next_token = torch.argmax(logits, dim=-1)
            
            t_token_end = time.time()
            token_gen_times.append(t_token_end - t_token_start)
            print(self.tokenizer.decode(next_token), end="", flush=True)
            current_pos += 1

        if token_gen_times:
            avg_time = sum(token_gen_times[1:]) / len(token_gen_times[1:])
            print(f"\n\n--- Performance Stats[Edge] ---")
            print(f"Total Tokens: {len(token_gen_times)}")
            print(f"Average TPOT: {avg_time*1000:.2f} ms/token")
            print(f"Average Tokens/Sec: {1/avg_time:.2f} tokens/s")
            print(f"Time to First Token: {time_to_first_token*1000:.2f} ms")

    def process_single_request(self, req):
        """处理单个请求的完整流程 (Prefill + Decode)"""
        print(f"\n>>> [Edge] Processing Request {req.req_id} (Waited {time.time() - req.arrival_time:.3f}s)")
        
        # tokens = req.input_tokens.to(DEVICE)
        prompt = "The capital of France is"
        tokens = torch.tensor(self.tokenizer.encode(prompt), device=DEVICE).unsqueeze(0)

        seq_len = tokens.shape[1]
        
        # --- Phase 1: Prefill ---
        t_start = time.time()
        
        h = torch.nn.functional.embedding(tokens, self.weights["tok_embeddings.weight"]).to(torch.bfloat16)
        
        # 逐层计算 Prefill (KV 存入 Buffer)
        for layer in range(self.config.n_layers):
            h = self.forward_layer_prefill(h, layer, seq_len)
        
        # [修改 7] 计算完成后，统一传输 KV Cache
        self.send_buffered_kv()

        # Prefill 结束，计算第一个 token
        h_final = rms_norm(h, self.weights["norm.weight"], self.config.norm_eps)
        logits = torch.matmul(h_final[:, -1, :], self.weights["output.weight"].T)
        next_token = torch.argmax(logits, dim=-1)
        
        ttft = time.time() - t_start # Time To First Token
        print(f"[Edge] Req {req.req_id} Prefill Done. TTFT: {ttft*1000:.2f} ms")

        # --- Phase 2: Decoding ---
        current_pos = seq_len
        generated_tokens = []
        
        t_decode_start = time.time()
        
        for _ in range(req.output_len):
            token_tensor = torch.tensor([[next_token]], device=DEVICE)
            h = torch.nn.functional.embedding(token_tensor, self.weights["tok_embeddings.weight"]).to(torch.bfloat16)
            
            # Local Head Layers
            for layer in range(0, CLOUD_START_LAYER):
                h = self.forward_layer_decode(h, layer, current_pos)
            
            # Remote Body Layers (RPC)
            req_payload = {'h': h, 'start_pos': current_pos}
            req_bytes = serialize_tensor(req_payload)
            
            with self.send_lock:
                send_packet(self.sock, MSG_HIDDEN_REQ, req_bytes)
            
            _, _, res_bytes = recv_packet(self.sock)
            h = deserialize_tensor(res_bytes)
            
            # Local Tail Layers
            for layer in range(CLOUD_END_LAYER, self.config.n_layers):
                h = self.forward_layer_decode(h, layer, current_pos)
            
            # Output Head
            h_final = rms_norm(h, self.weights["norm.weight"], self.config.norm_eps)
            logits = torch.matmul(h_final[:, -1, :], self.weights["output.weight"].T)
            next_token = torch.argmax(logits, dim=-1)
            
            generated_tokens.append(next_token.item())
            current_pos += 1

        total_time = time.time() - req.arrival_time # 包含排队时间
        inference_time = time.time() - t_start # 仅计算时间
        avg_tpot = (time.time() - t_decode_start) / req.output_len
        
        return {
            "req_id": req.req_id,
            "ttft": ttft,
            "tpot": avg_tpot,
            "total_latency": total_time,
            "queue_latency": t_start - req.arrival_time
        }

    def run_benchmark(self, arrival_rate=1.0, total_requests=10):
        """主循环：启动生成器并处理队列"""
        request_queue = queue.Queue()
        
        # 1. 启动负载生成器线程
        generator = WorkloadGenerator(request_queue, arrival_rate, total_requests)
        generator.start()
        
        stats = []
        processed_count = 0
        
        print(f"[System] Benchmark started. Rate={arrival_rate}, Total={total_requests}")
        
        try:
            while processed_count < total_requests:
                try:
                    req = request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # 处理请求
                result = self.process_single_request(req)
                stats.append(result)
                processed_count += 1
                
                print(f"[Result] Req {result['req_id']} Finished. "
                      f"Queue Wait: {result['queue_latency']*1000:.1f}ms, "
                      f"TTFT: {result['ttft']*1000:.1f}ms, "
                      f"TPOT: {result['tpot']*1000:.1f}ms")
                
                # 清理显存，防止多请求累积OOM
                # self.kv_cache.clear() 

        except KeyboardInterrupt:
            print("\n[System] Stopping benchmark...")
            generator.stop_event.set()
        
        generator.join()
        
        # --- 打印最终统计 ---
        print("\n" + "="*40)
        print(f"Benchmark Summary (Rate={arrival_rate} req/s)")
        print("="*40)
        avg_ttft = sum(r['ttft'] for r in stats) / len(stats)
        avg_tpot = sum(r['tpot'] for r in stats) / len(stats)
        avg_wait = sum(r['queue_latency'] for r in stats) / len(stats)
        
        print(f"Average TTFT       : {avg_ttft*1000:.2f} ms")
        print(f"Average TPOT       : {avg_tpot*1000:.2f} ms")
        print(f"Average Queue Wait : {avg_wait*1000:.2f} ms")
        print("="*40)

if __name__ == "__main__":
    client = EdgeClient()
    # 设为 1000 (极大值) 可以测试纯粹的吞吐量极限（无等待）
    client.run_benchmark(arrival_rate=1000, total_requests=5)