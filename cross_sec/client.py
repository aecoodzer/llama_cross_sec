import socket
import torch
import threading
import time
import queue
import os
import struct
import io

import numpy as np
from itertools import count
from transformers import AutoTokenizer
from utils import * 

# --- 全局配置 ---
DEVICE = get_device_edge() # 从 utils 获取设备，例如 "cuda:0"

class Request:
    def __init__(self, req_id, input_tokens, output_len, arrival_time):
        self.req_id = req_id
        self.input_tokens = input_tokens
        self.output_len = output_len
        self.arrival_time = arrival_time

class EdgeClient:
    def __init__(self):
        print(f"[Edge] Initializing on {DEVICE}...")
        self.config = LlamaConfig()
        # 1. 加载权重
        self.load_weights()
        # 2. 初始化 RoPE
        self.init_rope()
        # 3. 初始化 KV Cache 存储
        self.kv_cache = {} 

        # [新增] 预分配接收 Cloud 结果的 Buffer
        # Cloud 返回的也是 [1, 1, dim]
        self.cloud_result_buffer = torch.zeros(
            (1, 1, self.config.dim), 
            dtype=torch.bfloat16, 
            device=DEVICE
        )
        
        # 4. 网络连接
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 关键优化: 禁用 Nagle 算法，降低小包(Hidden State)延迟
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            self.sock.connect((SERVER_HOST, SERVER_PORT))
            print(f"[Edge] Connected to Cloud Server at {SERVER_HOST}:{SERVER_PORT}")
        except Exception as e:
            print(f"[Edge] Failed to connect: {e}")
            exit(1)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

        # 5. 序列化专用队列与线程
        # 队列元素: (cpu_kv_dict, cuda_event, layer_idx)
        self.serializer_queue = queue.Queue()
        self.serialize_thread = threading.Thread(target=self._serializer_worker, daemon=True)
        self.serialize_thread.start()

        # 6. 发送队列与守护线程
        # 队列元素: (priority, counter, msg_type, data_bytes, layer_idx)
        # priority: 1=High (HiddenReq), 2=Low (KVCache)
        self.send_queue = queue.PriorityQueue()
        self.running = True
        
        # 启动后台发送线程
        self.sender_thread = threading.Thread(target=self._network_sender, daemon=True)
        self.sender_thread.start()

        # 是否开启日志
        self.verbose = False



    def get_layer_cache(self, layer_idx):
        if layer_idx not in self.kv_cache:
            # 预分配形状: [Batch=1, MaxSeq, n_kv_heads, head_dim]
            # 注意：这里采用了 Channel Last 的布局 (Seq, Head)，避免 transpose
            # 也可以根据你的 xk 形状调整
            cache_shape = (1, MAX_SEQ_LEN, self.config.n_kv_heads, self.config.head_dim)
            
            self.kv_cache[layer_idx] = {
                'k': torch.zeros(cache_shape, dtype=torch.bfloat16, device=DEVICE),
                'v': torch.zeros(cache_shape, dtype=torch.bfloat16, device=DEVICE)
            }
        return self.kv_cache[layer_idx]

    def load_weights(self):
        print("[Edge] Loading weights...")
        # 加载全量权重用于 Prefill
        full_weights = torch.load(
            os.path.join(MODEL_PATH, "original/consolidated.00.pth"), 
            map_location=DEVICE, weights_only=True
        )
        self.weights = {k: v.to(torch.bfloat16) for k, v in full_weights.items()}
        print("[Edge] Weights loaded.")

    def init_rope(self):
        max_seq_len = 8192
        freqs = 1.0 / (self.config.rope_theta ** (torch.arange(0, self.config.head_dim, 2, device=DEVICE).float() / self.config.head_dim))
        t = torch.arange(max_seq_len, device=DEVICE, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    def _serializer_worker(self):
        """
        [后台线程] 等待 GPU 拷贝完成 -> CPU 序列化 -> 放入发送队列
        """
        print("[Edge] Serializer thread started.")
        while True:
            try:
                item = self.serializer_queue.get()
                if item is None: break
                
                cpu_kv, cuda_event, layer_idx = item
                
                # 等待 GPU->CPU 拷贝完成
                # 阻塞后台线程，主线程已经去算下一层了
                cuda_event.synchronize()
                
                # 2. CPU 密集型操作：序列化
                buffer = io.BytesIO()
                torch.save(cpu_kv, buffer)
                kv_bytes = buffer.getvalue()
                
                # 3. 放入网络发送队列
                # 优先级 2 (低优先), 记录时间戳
                self.send_queue.put((2, time.time(), MSG_KV_CACHE, kv_bytes, layer_idx))
                
            except Exception as e:
                print(f"[Serializer] Error: {e}")

    def _network_sender(self):
        """
        后台守护线程：专门负责从队列取数据并发送。
        PriorityQueue 会确保优先级高(数值小)的任务先被取出。
        """
        print("[Edge] Sender thread started.")
        while self.running:
            try:
                # 阻塞获取任务
                priority, _, msg_type, data_bytes, layer_idx = self.send_queue.get()
                
                # 发送数据
                # send_packet 应在 utils 中定义，负责封装 Header + Body
                send_packet(self.sock, msg_type, data_bytes, layer_idx)
                
                # 标记任务完成
                self.send_queue.task_done()
                
            except Exception as e:
                print(f"[Edge] Sender Error: {e}")
                break

    def forward_layer_prefill(self, h, layer_idx, seq_len):
        """
        Prefill 阶段单层前向传播
        功能：计算 Output + 提取并异步发送 Cloud 层的 KV Cache
        """
        # --- 1. RMS Norm ---
        norm_w = self.weights[f"layers.{layer_idx}.attention_norm.weight"]
        h_norm = rms_norm(h, norm_w, self.config.norm_eps)

        # --- 2. QKV Projection ---
        wq = self.weights[f"layers.{layer_idx}.attention.wq.weight"]
        wk = self.weights[f"layers.{layer_idx}.attention.wk.weight"]
        wv = self.weights[f"layers.{layer_idx}.attention.wv.weight"]
        wo = self.weights[f"layers.{layer_idx}.attention.wo.weight"]

        xq = torch.matmul(h_norm, wq.T).view(1, seq_len, self.config.n_heads, self.config.head_dim)
        xk = torch.matmul(h_norm, wk.T).view(1, seq_len, self.config.n_kv_heads, self.config.head_dim)
        xv = torch.matmul(h_norm, wv.T).view(1, seq_len, self.config.n_kv_heads, self.config.head_dim)

        # --- 3. RoPE ---
        freqs_cis_curr = self.freqs_cis[0 : seq_len].view(1, seq_len, 1, -1)
        xq = torch.view_as_real(torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) * freqs_cis_curr).flatten(3).to(torch.bfloat16)
        xk = torch.view_as_real(torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) * freqs_cis_curr).flatten(3).to(torch.bfloat16)

        # --- 4. KV Cache 处理 (核心逻辑) ---
        # 必须 detach 切断梯度，contiguous 保证内存连续
        current_k = xk.detach()
        current_v = xv.detach()

        # 判断是否属于 Cloud 负责的层
        if CLOUD_START_LAYER <= layer_idx < CLOUD_END_LAYER:
            # 1. 申请锁页内存 (Pinned Memory)，允许 GPU DMA 直接写入
            cpu_k = torch.empty(current_k.shape, dtype=current_k.dtype, device="cpu", pin_memory=True)
            cpu_v = torch.empty(current_v.shape, dtype=current_v.dtype, device="cpu", pin_memory=True)

            # 2. 异步拷贝 (Non-blocking)
            # 这两行代码瞬间返回，不会等待数据传完
            cpu_k.copy_(current_k, non_blocking=True)
            cpu_v.copy_(current_v, non_blocking=True)

            # 3. 记录 CUDA 事件 (作为同步路标)
            event = torch.cuda.Event()
            event.record()

            # 4. 扔给后台线程，主线程立刻返回去算下一层
            self.serializer_queue.put(({'k': cpu_k, 'v': cpu_v}, event, layer_idx))
        else:
            cache = self.get_layer_cache(layer_idx)
            # 使用切片写入，零拷贝，不申请新内存
            # 检查是否越界
            if seq_len > MAX_SEQ_LEN:
                raise RuntimeError(f"Input length {seq_len} exceeds limit {MAX_SEQ_LEN}")
            # 本地层，存入本地内存
            # 写入 Buffer 的 [0 : seq_len] 位置
            cache['k'][:, 0:seq_len, :, :] = current_k
            cache['v'][:, 0:seq_len, :, :] = current_v
        
        # --- 5. Attention Computation ---
        xq = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)

        # GQA 处理
        if self.config.n_kv_heads < self.config.n_heads:
            n_rep = self.config.n_heads // self.config.n_kv_heads
            keys = keys.repeat_interleave(n_rep, dim=1)
            values = values.repeat_interleave(n_rep, dim=1)

        scores = torch.matmul(xq, keys.transpose(-2, -1)) / (self.config.head_dim ** 0.5)
        
        # Causal Mask (Prefill 阶段需要)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=DEVICE)
        mask = torch.triu(mask, diagonal=1)
        scores = scores + mask
        
        probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        output = torch.matmul(probs, values).transpose(1, 2).contiguous().reshape(1, seq_len, self.config.dim)
        
        h_out = h + torch.matmul(output, wo.T)

        # --- 6. Feed Forward (SwiGLU) ---
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
        """
        Decode 阶段单层前向传播 (优化版：静态显存预分配)
        功能：读取本地 KV Cache，更新 Cache (零拷贝)，计算单 Token 输出
        """
        seq_len = 1
        
        # --- 1. 获取预分配的 Cache Buffer ---
        # 注意：这里只应该被 Edge 负责的层调用
        # get_layer_cache 负责返回预先申请好的 [Batch, MaxSeq, Head, Dim] 大张量
        cache = self.get_layer_cache(layer_idx) 
        k_buffer = cache['k']
        v_buffer = cache['v']

        # 越界检查 (防止显存溢出)
        if start_pos + seq_len > MAX_SEQ_LEN:
             raise RuntimeError(f"Decode pos {start_pos} exceeds limit {MAX_SEQ_LEN}")

        # --- 2. RMS Norm & QKV Projection ---
        # 这一步计算出当前 Token 的 Query, Key, Value
        norm_w = self.weights[f"layers.{layer_idx}.attention_norm.weight"]
        h_norm = rms_norm(h, norm_w, self.config.norm_eps)

        wq = self.weights[f"layers.{layer_idx}.attention.wq.weight"]
        wk = self.weights[f"layers.{layer_idx}.attention.wk.weight"]
        wv = self.weights[f"layers.{layer_idx}.attention.wv.weight"]
        wo = self.weights[f"layers.{layer_idx}.attention.wo.weight"]

        xq = torch.matmul(h_norm, wq.T).view(1, seq_len, self.config.n_heads, self.config.head_dim)
        xk = torch.matmul(h_norm, wk.T).view(1, seq_len, self.config.n_kv_heads, self.config.head_dim)
        xv = torch.matmul(h_norm, wv.T).view(1, seq_len, self.config.n_kv_heads, self.config.head_dim)

        # --- 3. RoPE (Rotary Positional Embedding) ---
        # 根据当前绝对位置 start_pos 获取旋转编码并应用
        freqs_cis_curr = self.freqs_cis[start_pos : start_pos + seq_len].view(1, seq_len, 1, -1)
        xq = torch.view_as_real(torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) * freqs_cis_curr).flatten(3).to(torch.bfloat16)
        xk = torch.view_as_real(torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) * freqs_cis_curr).flatten(3).to(torch.bfloat16)

        # --- 4. Update Cache (关键修改：In-place Update) ---
        # 将当前 token 的 KV 直接填入固定 Buffer 的指定位置
        # 注意：必须 detach()，否则计算图会随着时间步无限增长，导致显存泄漏
        k_buffer[:, start_pos : start_pos+1, :, :] = xk.detach()
        v_buffer[:, start_pos : start_pos+1, :, :] = xv.detach()

        # --- 5. 获取有效历史数据 (关键修改：Slicing) ---
        # 使用切片操作 (View)，不会发生内存拷贝，直接“指向”Buffer中 [0, start_pos] 的有效数据
        keys = k_buffer[:, :start_pos+1, :, :]
        values = v_buffer[:, :start_pos+1, :, :]

        # --- 6. Attention Computation ---
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # GQA (Grouped Query Attention) 重复 Key/Value
        if self.config.n_kv_heads < self.config.n_heads:
            n_rep = self.config.n_heads // self.config.n_kv_heads
            keys = keys.repeat_interleave(n_rep, dim=1)
            values = values.repeat_interleave(n_rep, dim=1)
        
        # 计算 Attention Scores
        # Decode 阶段不需要 Causal Mask，因为我们只取了过去时刻的 keys/values
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / (self.config.head_dim ** 0.5)
        probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        output = torch.matmul(probs, values).transpose(1, 2).reshape(1, seq_len, self.config.dim)
        
        h_out = h + torch.matmul(output, wo.T)

        # --- 7. FFN (Feed Forward Network) ---
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

    def run_request(self, req):
        """
        执行一个请求（用于负载测试）
        返回:
        ttft_s: prefill 阶段耗时（秒）
        token_gen_times: decode 阶段每个 token 的耗时列表（秒）
        """
        # ===== 输入 =====

        self.kv_cache.clear()
        tokens = req.input_tokens.to(DEVICE)
        seq_len = tokens.shape[1]
        out_len = req.output_len  # 用请求自带的输出长度（128）

        if getattr(self, "verbose", False):
            print(f"\n[Client] run_request req_id={req.req_id} prompt_len={seq_len} out_len={out_len}")

        # ================= Phase 1: Prefill =================
        t0 = time.time()
        if getattr(self, "verbose", False):
            print("[Edge] Starting Prefill (Pipeline Sending)...")
        # embedding
        h = torch.nn.functional.embedding(tokens, self.weights["tok_embeddings.weight"]).to(torch.bfloat16)
        # 逐层 prefill（会触发 cloud 层 KV 的异步发送）
        for layer in range(self.config.n_layers):
            h = self.forward_layer_prefill(h, layer, seq_len)

        # 计算第一个 next_token
        h_final = rms_norm(h, self.weights["norm.weight"], self.config.norm_eps)
        logits = torch.matmul(h_final[:, -1, :], self.weights["output.weight"].T)
        next_token = torch.argmax(logits, dim=-1)

        ttft_s = time.time() - t0
        if getattr(self, "verbose", False):
            print(f"[Edge] Prefill Done. TTFT: {ttft_s*1000:.2f} ms")
        # ================= Phase 2: Collaborative Decoding =================
        if getattr(self, "verbose", False):
            print("[Edge] Starting Collaborative Decoding...")
        current_pos = seq_len
        token_gen_times = []

        # 用于 PriorityQueue 同优先级时排序（你已有这个写法）
        req_counter = count()

        for _ in range(out_len):
            t_token_start = time.time()

            # 1) Embedding（单 token）
            token_tensor = torch.tensor([[next_token.item()]], device=DEVICE)  # 注意 item()，避免 shape/类型怪问题
            h = torch.nn.functional.embedding(token_tensor, self.weights["tok_embeddings.weight"]).to(torch.bfloat16)

            # 2) Edge head layers
            for layer in range(0, CLOUD_START_LAYER):
                h = self.forward_layer_decode(h, layer, current_pos)

            # 3) Cloud body layers（RPC）
            req_payload = {'h': h, 'start_pos': current_pos}
            req_bytes = serialize_tensor(req_payload)

            # 发请求：高优先级
            self.send_queue.put((1, next(req_counter), MSG_HIDDEN_REQ, req_bytes, -1))

            # 收响应：阻塞等待（单 worker OK，多 worker 会串包，需要 req_id + receiver）
            msg_type, layer_idx, res_bytes = recv_packet(self.sock)
            if not res_bytes:
                raise RuntimeError("Server disconnected during decode.")

            # server 发送的是 serialize_tensor(h)（纯 tensor）
            h = deserialize_tensor(res_bytes, DEVICE)

            # 4) Edge tail layers
            for layer in range(CLOUD_END_LAYER, self.config.n_layers):
                h = self.forward_layer_decode(h, layer, current_pos)

            # 5) Output head -> next_token
            h_final = rms_norm(h, self.weights["norm.weight"], self.config.norm_eps)
            logits = torch.matmul(h_final[:, -1, :], self.weights["output.weight"].T)
            next_token = torch.argmax(logits, dim=-1)

            # 6) 输出 token(查看推理正确性，应该只在验证模式下打印)
            print(self.tokenizer.decode(next_token), end="", flush=True)
            # 计时
            token_gen_times.append(time.time() - t_token_start)
            current_pos += 1

        return ttft_s, token_gen_times

        # 你可以选择不在这里打印，避免多请求时刷屏
        if getattr(self, "verbose", False) and token_gen_times:
            valid = token_gen_times[1:] if len(token_gen_times) > 1 else token_gen_times
            avg_tpot = sum(valid) / len(valid)
            print(f"--- Req {req.req_id} Stats ---")
            print(f"Tokens Generated: {len(token_gen_times)}")
            print(f"Avg TPOT: {avg_tpot*1000:.2f} ms")
            print(f"Throughput: {1/avg_tpot:.2f} tokens/s")

        # ⚠️ 不要在这里 self.running = False
        # 否则一个请求结束就把 sender 线程停了，后续请求发不出去

        return ttft_s, token_gen_times
    

def verify_correctness(client):
    prompt_text = "The capital of France is"
    model_path = MODEL_PATH
    print("\n" + "="*40)
    print(f" >>> 功能一：正确性验证 (Prompt: {prompt_text[:20]}...)")
    print("="*40)
    
    # 1. 临时开启详细日志
    client.verbose = True
    
    # 2. Tokenizer (使用 transformers 加载)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except:
        print("[Warning] Tokenizer load failed, utilizing dummy tokens.")
        tokenizer = None

    # 3. 构造 Input Tensor
    if tokenizer:
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    else:
        input_ids = torch.randint(0, 32000, (1, 10)).long() # Mock

    # 4. 构造 Request
    req = Request(req_id="verify_001", input_tokens=input_ids, output_len=128, arrival_time=time.time())
    
    # 5. 运行推理
    ttft, token_times = client.run_request(req)
    
    print(f"[Verify] Completed. TTFT: {ttft*1000:.2f} ms, Avg TPOT: {np.mean(token_times)*1000:.4f} ms")
    client.verbose = False # 恢复默认日志级别

# ==========================================
# 功能二：泊松分布负载测试 (Benchmark)
# ==========================================
class PoissonWorkloadGenerator:
    def __init__(self, arrival_rate, mean_input_len=4096, mean_output_len=128):
        self.rate = arrival_rate
        self.mean_in = mean_input_len
        self.mean_out = mean_output_len
        
    def generate_trace(self, duration_seconds):
        """生成整个测试周期的请求到达时间表"""
        requests = []
        t_current = 0
        req_id = 0
        
        while t_current < duration_seconds:
            # 泊松过程：间隔时间服从指数分布
            interval = np.random.exponential(scale=1.0/self.rate)
            t_current += interval
            
            # 构造合成数据
            # 长度可以稍微抖动一下，这里简化为固定或正态分布
            # in_len = int(np.random.normal(self.mean_in, 100)) 
            # out_len = int(np.random.normal(self.mean_out, 10))
            # 以上是正态分布，固定长度
            in_len = 4096
            out_len = 128
            in_len = max(1, in_len)
            out_len = max(1, out_len)
            
            # 随机生成 input tokens
            input_tokens = torch.randint(0, 32000, (1, in_len)).long()
            
            req = Request(req_id=f"req_{req_id}", 
                          input_tokens=input_tokens, 
                          output_len=out_len, 
                          arrival_time=t_current)
            requests.append(req)
            req_id += 1
            
        return requests
    
def run_benchmark(client, arrival_rate, duration=10):
    print("\n" + "="*60)
    print(f" >>> 功能二：负载测试 (Rate={arrival_rate} req/s, Duration={duration}s)")
    print("="*60)
    
    # 1. 生成请求轨迹
    workload = PoissonWorkloadGenerator(arrival_rate)
    requests = workload.generate_trace(duration)
    print(f"[Bench] Generated {len(requests)} requests based on Poisson process.")
    
    # 2. 共享队列 (模拟真实系统的请求池)
    req_queue = queue.Queue()
    results = [] # 存储结果 (req_id, ttft, avg_tpot, queuing_delay)
    
    # 3. 生产者线程：模拟按照时间戳“到达”
    def producer():
        t_start = time.time()
        for req in requests:
            # 计算需要等待的时间
            now = time.time() - t_start
            wait_time = req.arrival_time - now
            if wait_time > 0:
                time.sleep(wait_time)
            
            # 请求到达，放入处理队列
            req_queue.put(req)
            # print(f"[Arrive] {req.req_id} at {time.time()-t_start:.2f}s")
    
    prod_thread = threading.Thread(target=producer)
    prod_thread.start()
    
    # 4. 消费者 (主线程)：模拟 GPU 推理引擎处理
    # 我们的 EdgeClient 还是串行的，所以这里是 FIFO 处理
    processed_count = 0
    start_bench_time = time.time()
    
    while processed_count < len(requests):
        try:
            # 阻塞获取，模拟等待请求到达
            req = req_queue.get(timeout=2) 
            
            # 计算排队延迟：实际开始处理时间 - 理论到达时间
            actual_start_time = time.time() - start_bench_time
            queuing_delay = max(0, actual_start_time - req.arrival_time)
            
            # 执行推理
            ttft, token_times = client.run_request(req)
            avg_tpot = sum(token_times) / len(token_times) if token_times else 0
            
            results.append({
                "id": req.req_id,
                "ttft": ttft,
                "tpot": avg_tpot,
                "queue_delay": queuing_delay
            })
            
            processed_count += 1
            print(f"\r[Progress] {processed_count}/{len(requests)} | Last TTFT: {ttft*1000:.1f}ms", end="")
            
        except queue.Empty:
            # 如果生产者结束了且队列空了
            if not prod_thread.is_alive():
                break
    
    prod_thread.join()
    print("\n[Bench] Complete.")
    
    # 5. 统计分析
    ttfts = [r['ttft']*1000 for r in results]
    tpots = [r['tpot']*1000 for r in results]
    q_delays = [r['queue_delay']*1000 for r in results]
    
    print("-" * 30)
    print(f"Metric (Rate {arrival_rate}) |  Avg  |  P50  |  P99  ")
    print("-" * 30)
    print(f"TTFT (ms)          | {np.mean(ttfts):5.1f} | {np.percentile(ttfts, 50):5.1f} | {np.percentile(ttfts, 99):5.1f}")
    print(f"TPOT (ms)          | {np.mean(tpots):5.1f} | {np.percentile(tpots, 50):5.1f} | {np.percentile(tpots, 99):5.1f}")
    print(f"Queue Delay (ms)   | {np.mean(q_delays):5.1f} | {np.percentile(q_delays, 50):5.1f} | {np.percentile(q_delays, 99):5.1f}")
    print("-" * 30)

if __name__ == "__main__":
    client = EdgeClient()
    # client.run()
    verify_correctness(client)
    verify_correctness(client)

    # target_rates = [0.5, 1, 2, 4, 8]
    # # target_rates = [0.2]
    # for rate in target_rates:
    #     # 每次跑 10 秒钟的数据量
    #     run_benchmark(client, arrival_rate=rate, duration=10)
        
    #     # 冷却一下，防止日志混淆或发热
    #     time.sleep(1)