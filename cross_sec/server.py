import socket
import torch
import os
import time
import threading
import queue
from collections import defaultdict
from utils import *

DEVICE_CLOUD = get_device_cloud()

class CloudServer:
    def __init__(self):
        print(f"[Cloud] Initializing on {DEVICE_CLOUD}...")
        self.config = LlamaConfig()
        self.load_weights()
        self.init_rope()
        # ---引入同步原语 ---
        # 使用 threading.Event 替代 sleep 轮询
        # key: layer_idx, value: threading.Event
        self.layer_events = defaultdict(threading.Event)
        # 用于解耦 网络接收(IO) 和 模型计算(Compute)
        self.compute_queue = queue.Queue()
        self.kv_store = {} 

        # 反序列化专用队列
        # 元素: (msg_type, payload_bytes, layer_idx, conn)
        self.deserializer_queue = queue.Queue()
        # 2. 启动 Helper 线程 (负责 CPU反序列化 -> GPU搬运)
        self.helper_thread = threading.Thread(target=self._helper_worker, daemon=True)
        self.helper_thread.start()

    def load_weights(self):
        print("[Cloud] Loading weights...")
        # 保持原有逻辑，实际部署建议使用 safetensors 或 memmap 加速
        full_weights = torch.load(
            os.path.join(MODEL_PATH, "original/consolidated.00.pth"), 
            map_location=DEVICE_CLOUD, weights_only=True
        )
        self.weights = {k: v.to(torch.bfloat16) for k, v in full_weights.items()}
        print("[Cloud] Weights loaded.")

    def init_rope(self):
        # 保持原有 RoPE 逻辑
        max_seq_len = MAX_SEQ_LEN
        freqs = 1.0 / (self.config.rope_theta ** (torch.arange(0, self.config.head_dim, 2, device=DEVICE_CLOUD).float() / self.config.head_dim))
        t = torch.arange(max_seq_len, device=DEVICE_CLOUD, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    def get_layer_cache(self, layer_idx):
        if layer_idx not in self.kv_store:
            # 预分配显存 [Batch=1, MaxSeq, n_kv, head_dim]
            # 同样采用 Channel Last 布局方便切片
            cache_shape = (1, MAX_SEQ_LEN, self.config.n_kv_heads, self.config.head_dim)
            self.kv_store[layer_idx] = {
                'k': torch.zeros(cache_shape, dtype=torch.bfloat16, device=DEVICE_CLOUD),
                'v': torch.zeros(cache_shape, dtype=torch.bfloat16, device=DEVICE_CLOUD)
            }
        return self.kv_store[layer_idx]

    def _helper_worker(self):
        """
        [后台线程] 从队列取出二进制数据 -> CPU反序列化 -> 异步搬运到 GPU -> 更新状态
        """
        print("[Cloud] Helper thread started.")
        while True:
            # 阻塞获取任务
            item = self.deserializer_queue.get()
            if item is None: break
            
            msg_type, payload, layer_idx, conn = item
            
            try:
                # 1. 反序列化 (CPU 密集)
                # 注意：map_location 先设为 CPU，避免阻塞 IO 线程的数据接收
                buffer = io.BytesIO(payload)
                data = torch.load(buffer, map_location="cpu")
                
                if msg_type == MSG_KV_CACHE:
                    # 2. 搬运到 GPU (non_blocking)
                    inc_k = data['k'].to(DEVICE_CLOUD, non_blocking=True)
                    inc_v = data['v'].to(DEVICE_CLOUD, non_blocking=True)
                    
                    # 3. 填入显存 Buffer
                    self.update_kv_store_async(layer_idx, inc_k, inc_v)
                    
                elif msg_type == MSG_HIDDEN_REQ:
                    # 这里的 data 也是 CPU tensor
                    data['h'] = data['h'].to(DEVICE_CLOUD, non_blocking=True)
                    # 放入计算队列
                    self.compute_queue.put(data)
                    
            except Exception as e:
                print(f"[Helper] Error: {e}")

    def update_kv_store_async(self, layer_idx, k, v):
        """由 Helper 线程调用的 KV 更新函数"""
        cache = self.get_layer_cache(layer_idx)
        current_seq_len = k.shape[1]
        
        # 此时 k, v 已经在 GPU 上了
        cache['k'][:, 0:current_seq_len, :, :] = k
        cache['v'][:, 0:current_seq_len, :, :] = v
        
        # 激活事件，通知计算线程数据已就绪
        self.layer_events[layer_idx].set()

    def forward_layer(self, h, layer_idx, start_pos, seq_len):
        """
        Cloud 端只负责 Decode 阶段的计算
        """
        # 这里的 wait() 是阻塞直到 event 被 set()，没有 CPU 空转，且响应是微秒级的
        if layer_idx not in self.kv_store:
            # 如果当前层的 KV 还没到，进入操作系统级等待，不占用 GIL
            self.layer_events[layer_idx].wait()

        # 获取 Buffer
        cache = self.get_layer_cache(layer_idx)
        k_buffer = cache['k']
        v_buffer = cache['v']
        
        h_norm = rms_norm(h, self.weights[f"layers.{layer_idx}.attention_norm.weight"], self.config.norm_eps)
        
        wq = self.weights[f"layers.{layer_idx}.attention.wq.weight"]
        wk = self.weights[f"layers.{layer_idx}.attention.wk.weight"]
        wv = self.weights[f"layers.{layer_idx}.attention.wv.weight"]
        wo = self.weights[f"layers.{layer_idx}.attention.wo.weight"]
        
        xq = torch.matmul(h_norm, wq.T).view(1, seq_len, self.config.n_heads, self.config.head_dim)
        xk = torch.matmul(h_norm, wk.T).view(1, seq_len, self.config.n_kv_heads, self.config.head_dim)
        xv = torch.matmul(h_norm, wv.T).view(1, seq_len, self.config.n_kv_heads, self.config.head_dim)
        
        freqs_cis_curr = self.freqs_cis[start_pos : start_pos + seq_len].view(1, seq_len, 1, -1)
        xq = torch.view_as_real(torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) * freqs_cis_curr).flatten(3).to(torch.bfloat16)
        xk = torch.view_as_real(torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) * freqs_cis_curr).flatten(3).to(torch.bfloat16)

        # --- KV Cache 更新 ---
        # 越界检查
        if start_pos + seq_len > MAX_SEQ_LEN:
             raise RuntimeError(f"Cloud Layer {layer_idx} OOM: {start_pos}")
        
        # 写入新 Token 的 KV
        # 不需要 torch.cat
        k_buffer[:, start_pos : start_pos+seq_len, :, :] = xk
        v_buffer[:, start_pos : start_pos+seq_len, :, :] = xv
        
        # --- 读取有效历史---
        keys = k_buffer[:, :start_pos+seq_len, :, :]
        values = v_buffer[:, :start_pos+seq_len, :, :]
        
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
        
        h = h + torch.matmul(output, wo.T)
        
        h_norm = rms_norm(h, self.weights[f"layers.{layer_idx}.ffn_norm.weight"], self.config.norm_eps)
        w1 = self.weights[f"layers.{layer_idx}.feed_forward.w1.weight"]
        w2 = self.weights[f"layers.{layer_idx}.feed_forward.w2.weight"]
        w3 = self.weights[f"layers.{layer_idx}.feed_forward.w3.weight"]
        ffn_out = torch.matmul(torch.nn.functional.silu(torch.matmul(h_norm, w1.T)) * torch.matmul(h_norm, w3.T), w2.T)
        
        return h + ffn_out

    # --- IO 接收线程函数 ---
    def network_receiver(self, conn):
        print(f"[Cloud] IO Thread started for {conn.getpeername()}")
        try:
            while True:
                t0 = time.time()
                msg_type, layer_idx, payload = recv_packet(conn)
                t_network = time.time() - t0
                print(f"[Network] Client {id} recv latency: {t_network*1000:.2f}ms")
                
                if not payload: 
                    break

                # IO 线程只负责“收”，立刻扔进队列，不做任何 torch 操作
                self.deserializer_queue.put((msg_type, payload, layer_idx, conn))

        except Exception as e:
            print(f"[IO] Error: {e}")
        finally:
            self.compute_queue.put(None)
            print("[IO] Receiver thread exiting.")

    def start(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # 允许快速重启
        server_socket.bind((SERVER_HOST, SERVER_PORT))
        server_socket.listen(1)
        print(f"[Cloud] Listening on {SERVER_HOST}:{SERVER_PORT}")

        while True:
            print("[Cloud] Waiting for a new connection...")
            conn, addr = server_socket.accept()
            print(f"[Cloud] Connected by {addr}")
            
            # 重置状态
            self.kv_store = {}
            self.layer_events.clear() # 清除旧的 Events
            while not self.compute_queue.empty(): # 清空队列
                self.compute_queue.get()

            # --- 启动 IO 线程 ---
            io_thread = threading.Thread(target=self.network_receiver, args=(conn,))
            io_thread.daemon = True # 设置为守护线程
            io_thread.start()

            # ---主线程转变为计算 Worker ---
            try:
                while True:
                    # 从队列获取计算请求 (阻塞等待)
                    req_data = self.compute_queue.get()
                    
                    # 收到 None 表示连接断开或 IO 线程结束
                    if req_data is None:
                        break
                    
                    h = req_data['h']
                    start_pos = req_data['start_pos']
                    
                    t0 = time.time()
                    
                    # 流水线执行：如果某层的 KV 还没到，forward_layer 会自动 wait()
                    for layer in range(CLOUD_START_LAYER, CLOUD_END_LAYER):
                        h = self.forward_layer(h, layer, start_pos, seq_len=1)
                    
                    # 计算完成，发送结果
                    # 注意：socket 发送是线程安全的（Send vs Recv 分离），
                    # 但如果有多个计算线程同时 Send 则需要 Lock。这里只有一个计算线程，安全。
                    res_bytes = serialize_tensor(h)
                    send_packet(conn, MSG_HIDDEN_RES, res_bytes)
                    
                    # print(f"[Cloud] Compute finished in {(time.time()-t0)*1000:.2f}ms")

            except Exception as e:
                print(f"[Cloud] Compute loop error: {e}")
            finally:
                conn.close()
                print(f"[Cloud] Session with {addr} closed.")

if __name__ == "__main__":
    server = CloudServer()
    server.start()