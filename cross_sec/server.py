import socket
import torch
import os
import time
import threading
import queue
from collections import defaultdict
from utils import *

DEVICE_CLOUD = get_device_cloud()

# --- 客户端状态容器 ---
class ClientSession:
    """
    保存每个连接的独立状态：KV Cache、同步事件、Socket 连接
    """
    def __init__(self, conn, addr, client_id):
        self.conn = conn
        self.addr = addr
        self.client_id = client_id
        # 每个客户端独立的 KV 存储
        # key: layer_idx, value: {'k': tensor, 'v': tensor}
        self.kv_store = {}
        # 每个客户端独立的层级同步事件
        self.layer_events = defaultdict(threading.Event)
        self.active = True

    def get_layer_cache(self, config, layer_idx):
        if layer_idx not in self.kv_store:
            # 预分配显存 [Batch=1, MaxSeq, n_kv, head_dim]
            cache_shape = (1, MAX_SEQ_LEN, config.n_kv_heads, config.head_dim)
            self.kv_store[layer_idx] = {
                'k': torch.zeros(cache_shape, dtype=torch.bfloat16, device=DEVICE_CLOUD),
                'v': torch.zeros(cache_shape, dtype=torch.bfloat16, device=DEVICE_CLOUD)
            }
        return self.kv_store[layer_idx]

class CloudServer:
    def __init__(self):
        print(f"[Cloud] Initializing on {DEVICE_CLOUD}...")
        self.config = LlamaConfig()
        self.load_weights()
        self.init_rope()
        # # ---引入同步原语 ---
        # # 使用 threading.Event 替代 sleep 轮询
        # # key: layer_idx, value: threading.Event
        # self.layer_events = defaultdict(threading.Event)
        # # 用于解耦 网络接收(IO) 和 模型计算(Compute)
        # self.compute_queue = queue.Queue()
        # self.kv_store = {} 

        # # 反序列化专用队列
        # # 元素: (msg_type, payload_bytes, layer_idx, conn)
        # self.deserializer_queue = queue.Queue()
        # # 2. 启动 Helper 线程 (负责 CPU反序列化 -> GPU搬运)
        # self.helper_thread = threading.Thread(target=self._helper_worker, daemon=True)
        # self.helper_thread.start()

        # --- 核心竞争资源 ---
        # 全局计算队列：所有客户端的计算请求都汇聚于此
        # 元素格式: {'type': str, 'data': dict, 'session': ClientSession, 'arrival_time': float}
        self.global_compute_queue = queue.Queue()
        
        # 客户端列表 (用于管理)
        self.sessions = {} 
        self.lock = threading.Lock()

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


    # --- 独立的 GPU 工作线程 (Consumer) ---
    def gpu_worker_loop(self):
        print("[Cloud] GPU Worker Thread Started (Waiting for tasks...)")
        while True:
            # 1. 获取任务 (此处可能发生资源竞争，导致排队)
            task = self.global_compute_queue.get()
            
            # 计算排队延迟 (核心指标：Compute Bound 时该值会飙升)
            t_dequeue = time.time()
            queue_delay = (t_dequeue - task['arrival_time']) * 1000 # ms
            
            session = task['session']
            if not session.active:
                continue

            try:
                msg_type = task['type']
                payload = task['data'] # 已经是 CPU tensor (由接收线程反序列化)
                layer_idx = task.get('layer_idx', -1)

                if msg_type == MSG_KV_CACHE:
                    # --- 处理 KV 上传 ---
                    # 搬运 CPU -> GPU (耗时操作，但非计算密集)
                    k = payload['k'].to(DEVICE_CLOUD, non_blocking=True)
                    v = payload['v'].to(DEVICE_CLOUD, non_blocking=True)
                    
                    # 更新该 Client 的 KV Store
                    cache = session.get_layer_cache(self.config, layer_idx)
                    curr_len = k.shape[1]
                    cache['k'][:, 0:curr_len, :, :] = k
                    cache['v'][:, 0:curr_len, :, :] = v
                    
                    # 解锁该层的等待
                    session.layer_events[layer_idx].set()
                    
                    # print(f"[GPU] Client {session.client_id} Layer {layer_idx} KV Updated. Q-Delay: {queue_delay:.2f}ms")

                elif msg_type == MSG_HIDDEN_REQ:
                    # --- 处理推理请求 ---
                    h = payload['h'].to(DEVICE_CLOUD, non_blocking=True)
                    start_pos = payload['start_pos']
                    
                    # 如果排队时间过长，打印警告
                    if queue_delay > 10: 
                        print(f"[Contention] Client {session.client_id} Request Queued for {queue_delay:.2f}ms")

                    t_compute_start = time.time()

                    # 执行计算 (可能会因为等待 KV Cache 而阻塞，但那是逻辑依赖，不是资源竞争)
                    for layer in range(CLOUD_START_LAYER, CLOUD_END_LAYER):
                        h = self.forward_layer(h, layer, start_pos, 1, session)

                    # 发送回包 (IO 操作)
                    # 直接使用 session.conn 发送，socket 是线程安全的
                    res_bytes = serialize_tensor(h)
                    send_packet(session.conn, MSG_HIDDEN_RES, res_bytes)

                    # print(f"[GPU] Client {session.client_id} Compute Done. Cost: {(time.time()-t_compute_start)*1000:.2f}ms")

            except Exception as e:
                print(f"[GPU Error] Client {session.client_id}: {e}")
            finally:
                self.global_compute_queue.task_done()

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

    def forward_layer(self, h, layer_idx, start_pos, seq_len, session):
        """
        Cloud 端只负责 Decode 阶段的计算
        """

        if not session.layer_events[layer_idx].is_set():
            session.layer_events[layer_idx].wait()

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

    # --- 每个 Client 独立的接收线程 (Producer) ---
    def client_receiver_loop(self, session):
        client_id = session.client_id
        conn = session.conn
        print(f"[Cloud] IO Thread Started for Client {client_id}")
        
        try:
            while True:
                # 1. 网络接收 (Network Bound)
                # 多个客户端同时发送大量 KV Cache 时，这里会竞争带宽
                t_recv_start = time.time()
                msg_type, layer_idx, payload_bytes = recv_packet(conn)
                t_recv_end = time.time()
                
                if not payload_bytes:
                    break # 连接关闭

                # 记录网络耗时 (用于分析带宽瓶颈)
                recv_cost = (t_recv_end - t_recv_start) * 1000
                if recv_cost > 50: # 如果接收一个包超过 50ms，说明网络可能有拥塞
                    print(f"[Bandwidth] Client {client_id} recv large packet: {recv_cost:.2f}ms")

                # 2. 反序列化 (CPU Bound)
                # 在 IO 线程做反序列化，是为了让 GPU 线程只拿 Tensor
                # 这会消耗 CPU，如果有多个 Edge，CPU 可能成为瓶颈
                buffer = io.BytesIO(payload_bytes)
                data_tensor = torch.load(buffer, map_location="cpu") # 保持在 CPU

                # 3. 放入全局队列 (竞争锁)
                task = {
                    'type': msg_type,
                    'data': data_tensor,
                    'layer_idx': layer_idx,
                    'session': session,
                    'arrival_time': time.time() # 记录到达时间用于计算排队延迟
                }
                self.global_compute_queue.put(task)

        except Exception as e:
            print(f"[IO Error] Client {client_id}: {e}")
        finally:
            print(f"[Cloud] Client {client_id} Disconnected.")
            session.active = False
            conn.close()

    def start(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((SERVER_HOST, SERVER_PORT))
        server_socket.listen(5)
        print(f"[Cloud] Listening on {SERVER_HOST}:{SERVER_PORT}")

        # 启动唯一的 GPU 计算线程
        gpu_thread = threading.Thread(target=self.gpu_worker_loop, daemon=True)
        gpu_thread.start()

        client_count = 0
        while True:
            # 主循环只负责 Accept
            conn, addr = server_socket.accept()
            client_id = f"edge_{client_count}"
            client_count += 1
            print(f"[Cloud] Accepted connection from {addr}, assigned ID: {client_id}")

            # 创建 Session
            session = ClientSession(conn, addr, client_id)
            self.sessions[client_id] = session

            # 启动该 Client 的 IO 线程
            t = threading.Thread(target=self.client_receiver_loop, args=(session,), daemon=True)
            t.start()

if __name__ == "__main__":
    server = CloudServer()
    server.start()