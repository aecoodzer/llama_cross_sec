import socket
import torch
import threading
import time
from transformers import AutoTokenizer
from utils import *

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
class EdgeClient:
    def __init__(self):
        print(f"[Edge] Initializing on {DEVICE}...")
        self.config = LlamaConfig()
        self.load_weights()
        self.init_rope()
        self.kv_cache = {} 
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        # 连接 Cloud
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((SERVER_HOST, SERVER_PORT))
        self.send_lock = threading.Lock() # Socket 非线程安全，需要锁
        print(f"[Edge] Connected to Cloud Server.")

    def load_weights(self):
        # Edge 需要全量权重用于 Prefill 计算 (为了隐私，Edge 自己算所有层)
        # 但 Decode 时只用头尾层
        full_weights = torch.load(
            os.path.join(MODEL_PATH, "original/consolidated.00.pth"), 
            map_location=DEVICE, weights_only=True
        )
        self.weights = {k: v.to(torch.bfloat16) for k, v in full_weights.items()}

    def init_rope(self):
        # ... 同 Server ...
        max_seq_len = 4096
        freqs = 1.0 / (self.config.rope_theta ** (torch.arange(0, self.config.head_dim, 2, device=DEVICE).float() / self.config.head_dim))
        t = torch.arange(max_seq_len, device=DEVICE, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    def async_send_kv(self, layer_idx, kv_data):
        """线程函数：序列化并发送 KV"""
        kv_bytes = serialize_tensor(kv_data)
        with self.send_lock:
            send_packet(self.sock, MSG_KV_CACHE, kv_bytes, layer_idx)
        # print(f"[Edge-Thread] Sent KV Layer {layer_idx}")

    def forward_layer_prefill(self, h, layer_idx, seq_len):
        """
        Prefill 层的计算 + 触发异步发送
        包含完整的 RMSNorm, QKV Proj, RoPE, Causal Attention, FFN
        """
        # 1. --- RMS Norm & QKV Projection ---
        # 读取权重
        norm_w = self.weights[f"layers.{layer_idx}.attention_norm.weight"]
        wq = self.weights[f"layers.{layer_idx}.attention.wq.weight"]
        wk = self.weights[f"layers.{layer_idx}.attention.wk.weight"]
        wv = self.weights[f"layers.{layer_idx}.attention.wv.weight"]
        wo = self.weights[f"layers.{layer_idx}.attention.wo.weight"]

        # Norm
        h_norm = rms_norm(h, norm_w, self.config.norm_eps)

        # QKV Proj
        # Shape: [1, seq_len, n_heads, head_dim]
        xq = torch.matmul(h_norm, wq.T).view(1, seq_len, self.config.n_heads, self.config.head_dim)
        xk = torch.matmul(h_norm, wk.T).view(1, seq_len, self.config.n_kv_heads, self.config.head_dim)
        xv = torch.matmul(h_norm, wv.T).view(1, seq_len, self.config.n_kv_heads, self.config.head_dim)

        # 2. --- RoPE (Rotary Positional Embeddings) ---
        # 仅截取当前序列长度对应的频率
        freqs_cis_curr = self.freqs_cis[0 : seq_len].view(1, seq_len, 1, -1)
        
        # 应用旋转 (view_as_complex -> rotate -> view_as_real)
        # 注意：这里转 float32 计算以保证精度，最后转回 bfloat16
        xq = torch.view_as_real(torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) * freqs_cis_curr).flatten(3).to(torch.bfloat16)
        xk = torch.view_as_real(torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) * freqs_cis_curr).flatten(3).to(torch.bfloat16)

        # 3. --- KV Cache 处理 (核心：存本地 或 发送云端) ---
        # [重要优化] 使用 detach() 切断梯度计算图，使用 contiguous() 确保内存连续便于序列化
        k_cache = xk.detach().clone().contiguous()
        v_cache = xv.detach().clone().contiguous()
        current_kv = {'k': k_cache, 'v': v_cache}

        # 判断层归属：Cloud 负责的层异步发送，本地负责的层存入 cache
        if CLOUD_START_LAYER <= layer_idx < CLOUD_END_LAYER:
            # 启动线程发送，避免阻塞 GPU 计算主流
            t = threading.Thread(target=self.async_send_kv, args=(layer_idx, current_kv))
            t.start()
        else:
            # 本地层（如 Layer 0,1, 30,31）存入本地字典供 Decode 阶段使用
            self.kv_cache[layer_idx] = current_kv

        # 4. --- Attention Computation ---
        # 转换维度为 [batch, n_heads, seq_len, head_dim] 以进行矩阵乘法
        xq = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)

        # 处理 GQA (如果 KV heads 少于 Q heads，需要重复 KV)
        if self.config.n_kv_heads < self.config.n_heads:
            n_rep = self.config.n_heads // self.config.n_kv_heads
            keys = keys.repeat_interleave(n_rep, dim=1)
            values = values.repeat_interleave(n_rep, dim=1)

        # Scaled Dot-Product Attention
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / (self.config.head_dim ** 0.5)
        
        # Causal Mask (因果掩码)
        # 创建一个全 -inf 的矩阵，然后用 triu (上三角) 保留 -inf，其他位置置 0
        # 结果：下三角是 0 (保留注意力)，上三角是 -inf (屏蔽未来 token)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=DEVICE)
        mask = torch.triu(mask, diagonal=1)
        
        scores = scores + mask  # 广播加法
        probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        
        output = torch.matmul(probs, values)
        
        # 恢复形状 [1, seq_len, dim]
        output = output.transpose(1, 2).contiguous().reshape(1, seq_len, self.config.dim)

        # Output Projection + Residual Connection
        h_out = h + torch.matmul(output, wo.T)

        # 5. --- Feed Forward Network (FFN / SwiGLU) ---
        # Norm
        norm_ffn_w = self.weights[f"layers.{layer_idx}.ffn_norm.weight"]
        h_norm_ffn = rms_norm(h_out, norm_ffn_w, self.config.norm_eps)
        
        # Linear Projections
        w1 = self.weights[f"layers.{layer_idx}.feed_forward.w1.weight"]
        w2 = self.weights[f"layers.{layer_idx}.feed_forward.w2.weight"]
        w3 = self.weights[f"layers.{layer_idx}.feed_forward.w3.weight"]
        
        # SwiGLU: (SiLU(x * w1) * (x * w3)) * w2
        # F.silu is equivalent to Swish in this context
        ffn_out = torch.matmul(
            torch.nn.functional.silu(torch.matmul(h_norm_ffn, w1.T)) * torch.matmul(h_norm_ffn, w3.T), 
            w2.T
        )

        # Final Residual Connection
        return h_out + ffn_out

    def forward_layer_decode(self, h, layer_idx, start_pos):
        """Decode 阶段单层计算 (仅用于 Edge 自己的层)"""
        seq_len = 1
        
        # 1. 获取本地 KV Cache (在 Prefill 阶段已初始化)
        cache = self.kv_cache[layer_idx]
        
        # 2. RMS Norm & QKV Projection
        norm_w = self.weights[f"layers.{layer_idx}.attention_norm.weight"]
        h_norm = rms_norm(h, norm_w, self.config.norm_eps)
        
        wq, wk, wv, wo = [self.weights[f"layers.{layer_idx}.attention.{name}.weight"] for name in ['wq', 'wk', 'wv', 'wo']]
        
        xq = torch.matmul(h_norm, wq.T).view(1, seq_len, self.config.n_heads, self.config.head_dim)
        xk = torch.matmul(h_norm, wk.T).view(1, seq_len, self.config.n_kv_heads, self.config.head_dim)
        xv = torch.matmul(h_norm, wv.T).view(1, seq_len, self.config.n_kv_heads, self.config.head_dim)
        
        # 3. RoPE (Decode 阶段使用 start_pos)
        freqs_cis_curr = self.freqs_cis[start_pos : start_pos + seq_len].view(1, seq_len, 1, -1)
        xq = torch.view_as_real(torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) * freqs_cis_curr).flatten(3).to(torch.bfloat16)
        xk = torch.view_as_real(torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) * freqs_cis_curr).flatten(3).to(torch.bfloat16)
        
        # 4. Update KV Cache (拼接新 Token 的 KV)
        keys = torch.cat([cache['k'], xk], dim=1)
        values = torch.cat([cache['v'], xv], dim=1)
        self.kv_cache[layer_idx] = {'k': keys, 'v': values} # 更新本地缓存
        
        # 5. Attention
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # GQA / MQA Support
        if self.config.n_kv_heads < self.config.n_heads:
             n_rep = self.config.n_heads // self.config.n_kv_heads
             keys = keys.repeat_interleave(n_rep, dim=1)
             values = values.repeat_interleave(n_rep, dim=1)
        
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / (self.config.head_dim ** 0.5)
        probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        output = torch.matmul(probs, values).transpose(1, 2).reshape(1, seq_len, self.config.dim)
        
        h_out = h + torch.matmul(output, wo.T)
        
        # 6. FFN
        h_norm_ffn = rms_norm(h_out, self.weights[f"layers.{layer_idx}.ffn_norm.weight"], self.config.norm_eps)
        w1, w2, w3 = [self.weights[f"layers.{layer_idx}.feed_forward.{name}.weight"] for name in ['w1', 'w2', 'w3']]
        ffn_out = torch.matmul(torch.nn.functional.silu(torch.matmul(h_norm_ffn, w1.T)) * torch.matmul(h_norm_ffn, w3.T), w2.T)
        
        return h_out + ffn_out

    def run(self):
        prompt = "The capital of China is"
        print(f"\nPrompt: {prompt}")
        tokens = torch.tensor(self.tokenizer.encode(prompt), device=DEVICE).unsqueeze(0)
        seq_len = tokens.shape[1]
        
        # --- Phase 1: Prefill (Edge Compute ALL) ---
        print("[Edge] Starting Prefill (Pipeline Sending)...")
        t0 = time.time()
        
        h = torch.nn.functional.embedding(tokens, self.weights["tok_embeddings.weight"]).to(torch.bfloat16)
        
        for layer in range(self.config.n_layers):
            h = self.forward_layer_prefill(h, layer, seq_len)
        
        h_final = rms_norm(h, self.weights["norm.weight"], self.config.norm_eps)
        logits = torch.matmul(h_final[:, -1, :], self.weights["output.weight"].T)
        next_token = torch.argmax(logits, dim=-1)
        print(f"[Edge] Prefill Done. First token: {self.tokenizer.decode(next_token)}")
        
        # --- Phase 2: Collaborative Decoding ---
        print("[Edge] Starting Collaborative Decoding...")
        current_pos = seq_len
        
        for _ in range(10): # 生成 10 个词
            token_tensor = torch.tensor([[next_token]], device=DEVICE)
            h = torch.nn.functional.embedding(token_tensor, self.weights["tok_embeddings.weight"]).to(torch.bfloat16)
            
            # 1. Edge Compute Head Layers (0, 1)
            for layer in range(0, CLOUD_START_LAYER):
                h = self.forward_layer_decode(h, layer, current_pos)

            
            
            # 2. 发送 Hidden State 给 Cloud 计算 Body Layers (2-29)
            req_payload = {'h': h, 'start_pos': current_pos} # h 应该是经过 Layer 1 后的结果
            req_bytes = serialize_tensor(req_payload)
            
            t_net_start = time.time()
            with self.send_lock:
                send_packet(self.sock, MSG_HIDDEN_REQ, req_bytes)
            
            # 3. 等待 Cloud 返回
            _, _, res_bytes = recv_packet(self.sock)
            h = deserialize_tensor(res_bytes) # 这是 Layer 29 后的结果
            # print(f"  [RPC] Remote compute took {(time.time()-t_net_start)*1000:.2f}ms")
            
            # 4. Edge Compute Tail Layers (30, 31)
            for layer in range(CLOUD_END_LAYER, self.config.n_layers):
                h = self.forward_layer_decode(h, layer, current_pos)
                # pass
            
            # Output
            h_final = rms_norm(h, self.weights["norm.weight"], self.config.norm_eps)
            logits = torch.matmul(h_final[:, -1, :], self.weights["output.weight"].T)
            next_token = torch.argmax(logits, dim=-1)
            print(self.tokenizer.decode(next_token), end="", flush=True)
            current_pos += 1

if __name__ == "__main__":
    client = EdgeClient()
    client.run()