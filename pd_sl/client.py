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
        self.cloud_result_buffer = torch.zeros(
            (1, 1, self.config.dim), 
            dtype=torch.bfloat16, 
            device=DEVICE
        )
        
        # 4. 网络连接
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            self.sock.connect((SERVER_HOST, SERVER_PORT))
            print(f"[Edge] Connected to Cloud Server at {SERVER_HOST}:{SERVER_PORT}")
        except Exception as e:
            print(f"[Edge] Failed to connect: {e}")
            exit(1)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

        # 5. 序列化专用队列与线程
        # 【修改说明】：虽然保留了结构，但在Baseline模式下，我们将不在 forward 中使用它
        self.serializer_queue = queue.Queue()
        self.serialize_thread = threading.Thread(target=self._serializer_worker, daemon=True)
        self.serialize_thread.start()

        # 6. 发送队列与守护线程
        self.send_queue = queue.PriorityQueue()
        self.running = True
        
        self.sender_thread = threading.Thread(target=self._network_sender, daemon=True)
        self.sender_thread.start()

        self.verbose = False
        
        # 【新增】：用于在 Prefill 阶段暂存所有待发送的 KV 数据
        self.bulk_kv_buffer = []

    # ... (get_layer_cache, load_weights, init_rope, _serializer_worker, _network_sender 保持不变) ...
    # 为了节省篇幅，省略未修改的辅助函数，请直接保留原有的这部分代码
    
    
    
    def get_layer_cache(self, layer_idx):
        if layer_idx not in self.kv_cache:
            cache_shape = (1, MAX_SEQ_LEN, self.config.n_kv_heads, self.config.head_dim)
            self.kv_cache[layer_idx] = {
                'k': torch.zeros(cache_shape, dtype=torch.bfloat16, device=DEVICE),
                'v': torch.zeros(cache_shape, dtype=torch.bfloat16, device=DEVICE)
            }
        return self.kv_cache[layer_idx]

    def load_weights(self):
        print("[Edge] Loading weights...")
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
        # 原有的后台序列化线程，保持运行以防万一，但 Baseline 逻辑主要在主线程手动触发
        print("[Edge] Serializer thread started.")
        while True:
            try:
                item = self.serializer_queue.get()
                if item is None: break
                cpu_kv, cuda_event, layer_idx = item
                cuda_event.synchronize()
                buffer = io.BytesIO()
                torch.save(cpu_kv, buffer)
                kv_bytes = buffer.getvalue()
                self.send_queue.put((2, time.time(), MSG_KV_CACHE, kv_bytes, layer_idx))
            except Exception as e:
                print(f"[Serializer] Error: {e}")

    def _network_sender(self):
        print("[Edge] Sender thread started.")
        while self.running:
            try:
                priority, _, msg_type, data_bytes, layer_idx = self.send_queue.get()
                send_packet(self.sock, msg_type, data_bytes, layer_idx)
                self.send_queue.task_done()
            except Exception as e:
                print(f"[Edge] Sender Error: {e}")
                break

    def forward_layer_prefill(self, h, layer_idx, seq_len):
        """
        Prefill 阶段单层前向传播
        【修改说明】：取消异步发送，改为存入 self.bulk_kv_buffer
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

        # --- 4. KV Cache 处理 (Baseline 修改版) ---
        current_k = xk.detach()
        current_v = xv.detach()

        # 判断是否属于 Cloud 负责的层
        if CLOUD_START_LAYER <= layer_idx < CLOUD_END_LAYER:
            # 1. 依然申请锁页内存，保证拷贝速度
            cpu_k = torch.empty(current_k.shape, dtype=current_k.dtype, device="cpu", pin_memory=True)
            cpu_v = torch.empty(current_v.shape, dtype=current_v.dtype, device="cpu", pin_memory=True)

            # 2. 异步拷贝 (虽然是 Baseline，但 GPU->CPU 拷贝本身还是可以用 async 的，
            #    关键在于我们不会发出去，而是存起来)
            cpu_k.copy_(current_k, non_blocking=True)
            cpu_v.copy_(current_v, non_blocking=True)

            # 3. 记录事件
            event = torch.cuda.Event()
            event.record()

            # 【修改点】：不再放入 serializer_queue，而是放入本地暂存列表
            # 意味着直到 Prefill 结束前，数据都不会进入网络发送队列
            self.bulk_kv_buffer.append(({'k': cpu_k, 'v': cpu_v}, event, layer_idx))
        else:
            cache = self.get_layer_cache(layer_idx)
            if seq_len > MAX_SEQ_LEN:
                raise RuntimeError(f"Input length {seq_len} exceeds limit {MAX_SEQ_LEN}")
            cache['k'][:, 0:seq_len, :, :] = current_k
            cache['v'][:, 0:seq_len, :, :] = current_v
        
        # --- 5. Attention Computation ---
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
        output = torch.matmul(probs, values).transpose(1, 2).contiguous().reshape(1, seq_len, self.config.dim)
        
        h_out = h + torch.matmul(output, wo.T)

        # --- 6. Feed Forward ---
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

    # forward_layer_decode 保持不变，省略以节省篇幅 ...
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
        【修改说明】：
        1. 初始化 bulk_kv_buffer
        2. Prefill 循环中只计算，不发送
        3. Prefill 结束后，统一序列化并发送所有 KV Cache
        """
        # ===== 输入 =====
        self.kv_cache.clear()
        self.bulk_kv_buffer = [] # [新增] 清空暂存区

        tokens = req.input_tokens.to(DEVICE)
        seq_len = tokens.shape[1]
        out_len = req.output_len

        if getattr(self, "verbose", False):
            print(f"\n[Client] run_request req_id={req.req_id} prompt_len={seq_len} out_len={out_len}")

        # ================= Phase 1: Prefill =================
        t0 = time.time()
        if getattr(self, "verbose", False):
            print("[Edge] Starting Prefill (Wait-then-Send Mode)...")
        
        # embedding
        h = torch.nn.functional.embedding(tokens, self.weights["tok_embeddings.weight"]).to(torch.bfloat16)
        
        # 逐层 prefill
        for layer in range(self.config.n_layers):
            # 这里的 forward 已经被修改，只会 append 到 bulk_kv_buffer
            h = self.forward_layer_prefill(h, layer, seq_len)

        # 计算第一个 next_token (Edge 本地计算)
        h_final = rms_norm(h, self.weights["norm.weight"], self.config.norm_eps)
        logits = torch.matmul(h_final[:, -1, :], self.weights["output.weight"].T)
        next_token = torch.argmax(logits, dim=-1)

        # ================= Phase 1.5: Bulk Transmission (Baseline Implementation) =================
        # 【新增】：在这里集中处理 KV Cache 发送，模拟传统的 PD 分离逻辑
        # 只有当计算全部完成后，才开始传输
        
        t_prefill_compute_end = time.time()
        
        if len(self.bulk_kv_buffer) > 0:
            if getattr(self, "verbose", False):
                print(f"[Edge] Prefill Compute Done. Starting Bulk Transmission of {len(self.bulk_kv_buffer)} layers...")
            
            for item in self.bulk_kv_buffer:
                cpu_kv, cuda_event, layer_idx = item
                
                # 1. 确保 GPU->CPU 拷贝完成
                cuda_event.synchronize()
                
                # 2. 序列化 (CPU 密集型)
                buffer = io.BytesIO()
                torch.save(cpu_kv, buffer)
                kv_bytes = buffer.getvalue()
                
                # 3. 放入发送队列
                # 此时 send_queue 里的数据会以极快的速度堆积，然后由 sender_thread 发送
                # 为了更精确模拟“传输完成再 Decode”，我们其实应该等待 socket 发完，
                # 但由于 TCP 的流式特性，只要塞入 Kernel Buffer 就算发了。
                # 这里的阻塞主要体现在“序列化”的时间上。
                self.send_queue.put((2, time.time(), MSG_KV_CACHE, kv_bytes, layer_idx))
            
            # 清空 Buffer
            self.bulk_kv_buffer = []

        ttft_s = time.time() - t0 # 此时 TTFT 包含了 (计算 + 序列化 + 放入发送队列) 的时间
        if getattr(self, "verbose", False):
            print(f"[Edge] Prefill + Transfer Done. TTFT: {ttft_s*1000:.2f} ms")

        # ================= Phase 2: Collaborative Decoding =================
        if getattr(self, "verbose", False):
            print("[Edge] Starting Collaborative Decoding...")
        current_pos = seq_len
        token_gen_times = []
        req_counter = count()

        for _ in range(out_len):
            t_token_start = time.time()

            # 1) Embedding
            token_tensor = torch.tensor([[next_token.item()]], device=DEVICE)
            h = torch.nn.functional.embedding(token_tensor, self.weights["tok_embeddings.weight"]).to(torch.bfloat16)

            # 2) Edge head layers
            for layer in range(0, CLOUD_START_LAYER):
                h = self.forward_layer_decode(h, layer, current_pos)

            # 3) Cloud body layers (RPC)
            req_payload = {'h': h, 'start_pos': current_pos}
            req_bytes = serialize_tensor(req_payload)

            # 发请求
            self.send_queue.put((1, next(req_counter), MSG_HIDDEN_REQ, req_bytes, -1))

            # 收响应
            msg_type, layer_idx, res_bytes = recv_packet(self.sock)
            if not res_bytes:
                raise RuntimeError("Server disconnected during decode.")

            h = deserialize_tensor(res_bytes, DEVICE)

            # 4) Edge tail layers
            for layer in range(CLOUD_END_LAYER, self.config.n_layers):
                h = self.forward_layer_decode(h, layer, current_pos)

            # 5) Output
            h_final = rms_norm(h, self.weights["norm.weight"], self.config.norm_eps)
            logits = torch.matmul(h_final[:, -1, :], self.weights["output.weight"].T)
            next_token = torch.argmax(logits, dim=-1)

            # 6) Print & Record
            print(self.tokenizer.decode(next_token), end="", flush=True)
            token_gen_times.append(time.time() - t_token_start)
            current_pos += 1

        return ttft_s, token_gen_times