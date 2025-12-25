import socket
import torch
import os
import time
from utils import *

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
class CloudServer:
    def __init__(self):
        print(f"[Cloud] Initializing on {DEVICE}...")
        self.config = LlamaConfig()
        self.load_weights()
        self.init_rope()
        self.kv_store = {} # 存储接收到的 KV Cache: {layer_idx: {'k': ..., 'v': ...}}

    def load_weights(self):
        print("[Cloud] Loading weights...")
        # 实际部署只需加载 Layer 2-29。这里为了演示方便加载全量但只用部分。
        full_weights = torch.load(
            os.path.join(MODEL_PATH, "original/consolidated.00.pth"), 
            map_location=DEVICE, weights_only=True
        )
        self.weights = {k: v.to(torch.bfloat16) for k, v in full_weights.items()}
        print("[Cloud] Weights loaded.")

    def init_rope(self):
        # RoPE 预计算
        max_seq_len = 8192
        freqs = 1.0 / (self.config.rope_theta ** (torch.arange(0, self.config.head_dim, 2, device=DEVICE).float() / self.config.head_dim))
        t = torch.arange(max_seq_len, device=DEVICE, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    def forward_layer(self, h, layer_idx, start_pos, seq_len):
        """Cloud 端只负责 Decode 阶段的计算 (seq_len=1)"""
        # 1. 检查 KV Cache 是否就绪 (论文中的同步点)
        while layer_idx not in self.kv_store:
            # 简单的自旋等待，实际应用可用 Condition Variable
            time.sleep(0.001) 
        
        cache = self.kv_store[layer_idx]
        
        # --- 标准 Llama Attention 计算 ---
        h_norm = rms_norm(h, self.weights[f"layers.{layer_idx}.attention_norm.weight"], self.config.norm_eps)
        
        wq = self.weights[f"layers.{layer_idx}.attention.wq.weight"]
        wk = self.weights[f"layers.{layer_idx}.attention.wk.weight"]
        wv = self.weights[f"layers.{layer_idx}.attention.wv.weight"]
        wo = self.weights[f"layers.{layer_idx}.attention.wo.weight"]
        
        xq = torch.matmul(h_norm, wq.T).view(1, seq_len, self.config.n_heads, self.config.head_dim)
        xk = torch.matmul(h_norm, wk.T).view(1, seq_len, self.config.n_kv_heads, self.config.head_dim)
        xv = torch.matmul(h_norm, wv.T).view(1, seq_len, self.config.n_kv_heads, self.config.head_dim)
        
        # RoPE
        freqs_cis_curr = self.freqs_cis[start_pos : start_pos + seq_len].view(1, seq_len, 1, -1)
        xq = torch.view_as_real(torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) * freqs_cis_curr).flatten(3).to(torch.bfloat16)
        xk = torch.view_as_real(torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) * freqs_cis_curr).flatten(3).to(torch.bfloat16)

        # 更新 Cache (Decode 阶段追加)
        keys = torch.cat([cache['k'], xk], dim=1)
        values = torch.cat([cache['v'], xv], dim=1)
        self.kv_store[layer_idx] = {'k': keys, 'v': values}
        
        # Attention
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # GQA
        if self.config.n_kv_heads < self.config.n_heads:
            n_rep = self.config.n_heads // self.config.n_kv_heads
            keys = keys.repeat_interleave(n_rep, dim=1)
            values = values.repeat_interleave(n_rep, dim=1)
            
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / (self.config.head_dim ** 0.5)
        probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        output = torch.matmul(probs, values).transpose(1, 2).reshape(1, seq_len, self.config.dim)
        
        h = h + torch.matmul(output, wo.T)
        
        # FFN
        h_norm = rms_norm(h, self.weights[f"layers.{layer_idx}.ffn_norm.weight"], self.config.norm_eps)
        w1 = self.weights[f"layers.{layer_idx}.feed_forward.w1.weight"]
        w2 = self.weights[f"layers.{layer_idx}.feed_forward.w2.weight"]
        w3 = self.weights[f"layers.{layer_idx}.feed_forward.w3.weight"]
        ffn_out = torch.matmul(torch.nn.functional.silu(torch.matmul(h_norm, w1.T)) * torch.matmul(h_norm, w3.T), w2.T)
        
        return h + ffn_out

    def start(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((SERVER_HOST, SERVER_PORT))
        server_socket.listen(1)
        print(f"[Cloud] Listening on {SERVER_HOST}:{SERVER_PORT}")

        while True:
            print("[Cloud] Waiting for a new connection...")
            conn = None
            try:
                # 阻塞等待新的客户端连接
                conn, addr = server_socket.accept()
                print(f"[Cloud] Connected by {addr}")
                # 为新的会话清理 KV Cache，确保每轮推理从干净状态开始。
                self.kv_store = {}
                # 内层循环：处理当前客户端的所有请求 (KV Cache 发送和 Hidden State 请求)
                while True:
                    msg_type, layer_idx, payload = recv_packet(conn)
                    if not payload: break

                    if msg_type == MSG_KV_CACHE:
                        # 收到 Prefill 阶段的 KV Cache
                        kv_data = deserialize_tensor(payload)
                        self.kv_store[layer_idx] = kv_data
                        # print(f"[Cloud] Stored KV for Layer {layer_idx}")
                        
                    elif msg_type == MSG_HIDDEN_REQ:
                        # 收到 Decode 请求 (包含 start_pos 和 hidden_state)
                        # 协议稍微魔改一下，payload 存个 dict
                        req_data = deserialize_tensor(payload)
                        h = req_data['h']
                        start_pos = req_data['start_pos']
                            
                        # 连续计算 Cloud 负责的所有层
                        t0 = time.time()
                        for layer in range(CLOUD_START_LAYER, CLOUD_END_LAYER):
                            h = self.forward_layer(h, layer, start_pos, seq_len=1)
                            
                        # 返回结果
                        res_bytes = serialize_tensor(h)
                        send_packet(conn, MSG_HIDDEN_RES, res_bytes)
                        print(f"[Cloud] Processed layers {CLOUD_START_LAYER}-{CLOUD_END_LAYER} in {(time.time()-t0)*1000:.2f}ms")
            except ConnectionResetError:
                # 捕获客户端非正常断开连接的错误
                print(f"[Cloud] Connection reset by peer {addr}")
            except Exception as e:
                # 捕获其他处理错误
                print(f"[Cloud] Error processing request: {e}")
            finally:
                # 确保在异常或连接关闭时，当前连接被关闭
                if conn:
                    conn.close()
                # 外层 while True 循环将继续，等待下一个 accept()
if __name__ == "__main__":
    server = CloudServer()
    server.start()