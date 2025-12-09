import torch
import json
from transformers import AutoTokenizer
import os
import time

# --- 设备配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def rms_norm(tensor, norm_weights, norm_eps=1e-05):
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights

class LlamaInference:
    def __init__(self, model_path, config_path):
        self.device = DEVICE
        self.load_config(config_path)
        self.load_model(config_path)
        self.init_rope()
        self.kv_cache = [None] * self.n_layers
        
    def load_config(self, config_path):
        with open(os.path.join(config_path, "params.json"), "r") as f:
            config = json.load(f)
        self.dim = config["dim"]
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.n_kv_heads = config.get("n_kv_heads", self.n_heads)
        self.vocab_size = config["vocab_size"]
        self.norm_eps = config["norm_eps"]
        self.rope_theta = config.get("rope_theta", 500000.0)
        self.head_dim = self.dim // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads # GQA 重复次数

    def load_model(self, config_path):
        print("Loading model...")
        weights = torch.load(
            os.path.join(config_path, "consolidated.00.pth"), 
            map_location=self.device, 
            weights_only=True
        )
        # 转为 bfloat16
        self.weights = {k: v.to(torch.bfloat16) for k, v in weights.items()}
        print("Model loaded.")

    def init_rope(self):
        # 预计算 RoPE 旋转矩阵
        max_seq_len = 4096
        freqs = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, device=self.device).float() / self.head_dim))
        t = torch.arange(max_seq_len, device=self.device, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    def _forward_layer(self, h, layer_idx, start_pos, seq_len):
        """
        单层 Transformer 计算逻辑 (通用)
        """
        # 1. Norm
        h_norm = rms_norm(h, self.weights[f"layers.{layer_idx}.attention_norm.weight"], self.norm_eps)

        # 2. QKV Projection
        wq = self.weights[f"layers.{layer_idx}.attention.wq.weight"]
        wk = self.weights[f"layers.{layer_idx}.attention.wk.weight"]
        wv = self.weights[f"layers.{layer_idx}.attention.wv.weight"]
        wo = self.weights[f"layers.{layer_idx}.attention.wo.weight"]

        xq = torch.matmul(h_norm, wq.T).view(1, seq_len, self.n_heads, self.head_dim)
        xk = torch.matmul(h_norm, wk.T).view(1, seq_len, self.n_kv_heads, self.head_dim)
        xv = torch.matmul(h_norm, wv.T).view(1, seq_len, self.n_kv_heads, self.head_dim)

        # 3. RoPE
        freqs_cis_curr = self.freqs_cis[start_pos : start_pos + seq_len]
        freqs_cis_shaped = freqs_cis_curr.view(1, seq_len, 1, -1)
        
        xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        
        xq = torch.view_as_real(xq_c * freqs_cis_shaped).flatten(3).to(torch.bfloat16)
        xk = torch.view_as_real(xk_c * freqs_cis_shaped).flatten(3).to(torch.bfloat16)

        # 4. KV Cache Update (核心区别点)
        if start_pos == 0: 
            # Prefill 阶段：直接覆盖/新建 Cache
            keys, values = xk, xv
            self.kv_cache[layer_idx] = {'k': keys, 'v': values}
        else:
            # Decode 阶段：拼接 Cache
            current_cache = self.kv_cache[layer_idx]
            keys = torch.cat([current_cache['k'], xk], dim=1)
            values = torch.cat([current_cache['v'], xv], dim=1)
            self.kv_cache[layer_idx] = {'k': keys, 'v': values}

        # 5. Attention
        xq = xq.transpose(1, 2) # [1, n_heads, seq, dim]
        keys = keys.transpose(1, 2) # [1, kv_heads, total_seq, dim]
        values = values.transpose(1, 2)

        # GQA Expand
        if self.n_kv_heads < self.n_heads:
            keys = keys.repeat_interleave(self.n_rep, dim=1)
            values = values.repeat_interleave(self.n_rep, dim=1)

        scores = torch.matmul(xq, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Masking (仅在 Prefill 且 seq_len > 1 时需要)
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=self.device)
            mask = torch.triu(mask, diagonal=1)
            scores = scores + mask

        probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        output = torch.matmul(probs, values).transpose(1, 2).reshape(1, seq_len, self.dim)
        
        # 6. Output Projection & Residual
        h = h + torch.matmul(output, wo.T)

        # 7. FFN
        h_residual = h
        h_norm = rms_norm(h, self.weights[f"layers.{layer_idx}.ffn_norm.weight"], self.norm_eps)
        w1 = self.weights[f"layers.{layer_idx}.feed_forward.w1.weight"]
        w2 = self.weights[f"layers.{layer_idx}.feed_forward.w2.weight"]
        w3 = self.weights[f"layers.{layer_idx}.feed_forward.w3.weight"]
        
        ffn_out = torch.matmul(
            torch.nn.functional.silu(torch.matmul(h_norm, w1.T)) * torch.matmul(h_norm, w3.T), 
            w2.T
        )
        return h_residual + ffn_out

    def prefill(self, prompt_tokens):
        """
        Phase 1: Prefill
        处理完整的 Prompt，填充 KV Cache，并返回最后一个 token 的预测结果
        """
        print("[System] Starting Prefill...")
        # 重置 KV Cache
        self.kv_cache = [None] * self.n_layers
        
        seq_len = prompt_tokens.shape[1]
        h = torch.nn.functional.embedding(prompt_tokens, self.weights["tok_embeddings.weight"]).to(torch.bfloat16)
        
        # 逐层计算
        for layer in range(self.n_layers):
            h = self._forward_layer(h, layer, start_pos=0, seq_len=seq_len)
            
        # Final Norm & Logits
        h_final = rms_norm(h, self.weights["norm.weight"], self.norm_eps)
        # 只取最后一个位置的输出进行预测
        logits = torch.matmul(h_final[:, -1, :], self.weights["output.weight"].T)
        next_token = torch.argmax(logits, dim=-1)
        
        return next_token.item(), seq_len

    def decode(self, token_id, start_pos):
        """
        Phase 2: Decode
        处理单个 Token，追加 KV Cache，返回下一个 Token
        """
        # 输入准备 [1, 1]
        token_tensor = torch.tensor([[token_id]], device=self.device)
        h = torch.nn.functional.embedding(token_tensor, self.weights["tok_embeddings.weight"]).to(torch.bfloat16)
        
        # 逐层计算 (start_pos 传入当前位置)
        for layer in range(self.n_layers):
            h = self._forward_layer(h, layer, start_pos=start_pos, seq_len=1)
            
        h_final = rms_norm(h, self.weights["norm.weight"], self.norm_eps)
        logits = torch.matmul(h_final[:, -1, :], self.weights["output.weight"].T)
        next_token = torch.argmax(logits, dim=-1)
        
        return next_token.item()

def main():
    # 路径配置
    model_path = "/workspace/code/modelscope/Llama-3-8B"
    config_path = "/workspace/code/modelscope/Llama-3-8B/original"
    
    if not os.path.exists(model_path):
        print("Path error.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    engine = LlamaInference(model_path, config_path)
    
    prompt = "The capital of China is"
    print(f"\nPrompt: {prompt}")
    
    # 1. 准备数据
    tokens = torch.tensor(tokenizer.encode(prompt), device=DEVICE).unsqueeze(0)
    
    # --- PD 分离执行流程 ---
    
    # Phase 1: Prefill
    t0 = time.time()
    next_token, current_pos = engine.prefill(tokens)
    t_prefill = time.time() - t0
    print(f"[System] Prefill done in {t_prefill:.4f}s. Next token: {tokenizer.decode(next_token)}")
    
    print("Generating:", end=" ", flush=True)
    print(tokenizer.decode(next_token), end="", flush=True)
    
    # Phase 2: Decode Loop
    max_new_tokens = 20
    t_start_decode = time.time()
    
    for _ in range(max_new_tokens - 1):
        next_token = engine.decode(next_token, start_pos=current_pos)
        print(tokenizer.decode(next_token), end="", flush=True)
        current_pos += 1
        
    t_total_decode = time.time() - t_start_decode
    print(f"\n\n[Stats] Decode Speed: {(max_new_tokens-1)/t_total_decode:.2f} tokens/s")

if __name__ == "__main__":
    main()  