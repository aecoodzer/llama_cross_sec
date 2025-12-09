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
        """Prefill 层的计算 + 触发异步发送"""
        # ... 标准计算逻辑 (同 Server，为了省篇幅简写) ...
        # (此处应包含完整的 Norm, QKV, RoPE, Attn, FFN 计算)
        # 假设我们得到了 new_k, new_v 和 output_h
        
        # --- 真实计算代码复用 ---
        norm_w = self.weights[f"layers.{layer_idx}.attention_norm.weight"]
        h_norm = rms_norm(h, norm_w, self.config.norm_eps)
        wq, wk, wv, wo = [self.weights[f"layers.{layer_idx}.attention.{name}.weight"] for name in ['wq', 'wk', 'wv', 'wo']]
        
        xq = torch.matmul(h_norm, wq.T).view(1, seq_len, self.config.n_heads, self.config.head_dim)
        xk = torch.matmul(h_norm, wk.T).view(1, seq_len, self.config.n_kv_heads, self.config.head_dim)
        xv = torch.matmul(h_norm, wv.T).view(1, seq_len, self.config.n_kv_heads, self.config.head_dim)
        
        freqs_cis_curr = self.freqs_cis[0 : seq_len].view(1, seq_len, 1, -1)
        xq = torch.view_as_real(torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) * freqs_cis_curr).flatten(3).to(torch.bfloat16)
        xk = torch.view_as_real(torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) * freqs_cis_curr).flatten(3).to(torch.bfloat16)
        
        # 保存本地 KV 用于 Decode 阶段的头尾层，或者发送给 Cloud
        current_kv = {'k': xk, 'v': xv}
        
        # === 核心逻辑：Overlap ===
        # 如果这一层属于 Cloud 负责 (Layer 2-29)，则异步发送
        if CLOUD_START_LAYER <= layer_idx < CLOUD_END_LAYER:
            t = threading.Thread(target=self.async_send_kv, args=(layer_idx, current_kv))
            t.start()
        else:
            # 本地负责的层，存本地
            self.kv_cache[layer_idx] = current_kv
            
        # Attention Masked
        xq = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)
        if self.config.n_kv_heads < self.config.n_heads:
             keys = keys.repeat_interleave(self.config.n_heads // self.config.n_kv_heads, dim=1)
             values = values.repeat_interleave(self.config.n_heads // self.config.n_kv_heads, dim=1)
        
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / (self.config.head_dim ** 0.5)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=DEVICE)
        mask = torch.triu(mask, diagonal=1)
        scores = scores + mask
        probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        output = torch.matmul(probs, values).transpose(1, 2).reshape(1, seq_len, self.config.dim)
        
        h_out = h + torch.matmul(output, wo.T)
        
        # FFN
        h_norm_ffn = rms_norm(h_out, self.weights[f"layers.{layer_idx}.ffn_norm.weight"], self.config.norm_eps)
        w1, w2, w3 = [self.weights[f"layers.{layer_idx}.feed_forward.{name}.weight"] for name in ['w1', 'w2', 'w3']]
        ffn_out = torch.matmul(torch.nn.functional.silu(torch.matmul(h_norm_ffn, w1.T)) * torch.matmul(h_norm_ffn, w3.T), w2.T)
        
        return h_out + ffn_out

    def forward_layer_decode(self, h, layer_idx, start_pos):
        """Decode 阶段单层计算 (仅用于 Edge 自己的层)"""
        # ... (此处省略重复的 Decode 逻辑，与 Server 端的 forward_layer 几乎一致) ...
        # 简写：使用本地 kv_cache
        cache = self.kv_cache[layer_idx]
        # ... 计算 QKV, RoPE, Update Cache, Attention, FFN ...
        # (为了代码简洁，请参考 server.py 的 forward_layer，只需改为读取 self.kv_cache)
        return h # 伪代码返回

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
                # 这里需要你需要补全 forward_layer_decode 的实现，或者复用 server 的逻辑
                # 为演示，这里假设已经算完
                pass 
            
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
                # h = self.forward_layer_decode(h, layer, current_pos)
                pass
            
            # Output
            h_final = rms_norm(h, self.weights["norm.weight"], self.config.norm_eps)
            logits = torch.matmul(h_final[:, -1, :], self.weights["output.weight"].T)
            next_token = torch.argmax(logits, dim=-1)
            print(self.tokenizer.decode(next_token), end="", flush=True)
            current_pos += 1

if __name__ == "__main__":
    client = EdgeClient()
    client.run()