import torch
import transformers
import json
from transformers import AutoTokenizer
import os
import time

# --- 设置设备 ---
# 自动检测是否有 GPU，否则回退到 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {DEVICE}")

def rms_norm(tensor, norm_weights, norm_eps=1e-05):
    # 保持维度计算，避免广播错误
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights

def main():
    model_path = "/workspace/code/modelscope/Llama-3-8B"
    config_path = "/workspace/code/modelscope/Llama-3-8B/original"
    
    if not os.path.exists(model_path):
        print(f"Warning: Path {model_path} not found.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    config_file = os.path.join(config_path, "params.json")
    with open(config_file, "r") as f:
        config = json.load(f)

    # --- 1. 加载模型到 GPU ---
    print("Loading model to GPU (bfloat16)...")
    # map_location=DEVICE 直接加载到显存
    # weights_only=True 是为了安全
    model_weights = torch.load(
        os.path.join(config_path, "consolidated.00.pth"), 
        map_location=DEVICE, 
        weights_only=True
    )
    
    # 将所有权重转换为 bfloat16 以加速并减少显存占用
    for k, v in model_weights.items():
        model_weights[k] = v.to(torch.bfloat16)
        
    print("Model loaded.")

    # 参数提取
    dim = config["dim"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    n_kv_heads = config.get("n_kv_heads", n_heads)
    vocab_size = config["vocab_size"]
    norm_eps = config["norm_eps"]
    rope_theta = config.get("rope_theta", 500000.0)
    
    # GQA 参数
    n_rep = n_heads // n_kv_heads 
    head_dim = dim // n_heads

    # --- 2. 预计算 RoPE (在 GPU 上) ---
    max_seq_len = 2048
    # 使用 float32 计算频率以保证精度，最后转回
    freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=DEVICE).float() / head_dim))
    t = torch.arange(max_seq_len, device=DEVICE, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # Complex float

    # --- 3. 准备输入 ---
    prompt = "The capital of China is"
    tokens = tokenizer.encode(prompt)
    # 将输入 tokens 移到 GPU
    tokens = torch.tensor(tokens, device=DEVICE).unsqueeze(0) 

    # --- KV Cache 初始化 ---
    # 预分配显存列表
    kv_cache = [None] * n_layers
    
    max_new_tokens = 20
    print(f"\nPrompt: {prompt}")
    print("Output:", end=" ", flush=True)
    
    start_pos = 0
    curr_tokens = tokens

    # --- 4. 计时开始 ---
    t0 = time.time()

    with torch.no_grad():
        for i in range(max_new_tokens):
            seq_len = curr_tokens.shape[1]
            
            # Embedding
            embedding_weight = model_weights["tok_embeddings.weight"]
            # [1, seq_len, dim]
            h = torch.nn.functional.embedding(curr_tokens, embedding_weight).to(torch.bfloat16)

            # 获取 RoPE 切片
            freqs_cis_curr = freqs_cis[start_pos : start_pos + seq_len]
            # 调整形状以支持广播: [1, seq_len, 1, head_dim/2]
            freqs_cis_curr_shaped = freqs_cis_curr.view(1, seq_len, 1, -1)

            for layer in range(n_layers):
                h_residual = h
                # Norm
                norm_w = model_weights[f"layers.{layer}.attention_norm.weight"]
                h_norm = rms_norm(h, norm_w, norm_eps)

                # --- QKV 投影 (全矩阵运算) ---
                wq = model_weights[f"layers.{layer}.attention.wq.weight"]
                wk = model_weights[f"layers.{layer}.attention.wk.weight"]
                wv = model_weights[f"layers.{layer}.attention.wv.weight"]
                wo = model_weights[f"layers.{layer}.attention.wo.weight"]

                # Linear Projections
                xq = torch.matmul(h_norm, wq.T).view(1, seq_len, n_heads, head_dim)
                xk = torch.matmul(h_norm, wk.T).view(1, seq_len, n_kv_heads, head_dim)
                xv = torch.matmul(h_norm, wv.T).view(1, seq_len, n_kv_heads, head_dim)

                # --- RoPE (向量化) ---
                # 将 Q, K 转为复数
                xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
                xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
                
                # 旋转
                xq_rotated = torch.view_as_real(xq_complex * freqs_cis_curr_shaped).flatten(3).to(torch.bfloat16)
                xk_rotated = torch.view_as_real(xk_complex * freqs_cis_curr_shaped).flatten(3).to(torch.bfloat16)

                # --- KV Cache Update ---
                if kv_cache[layer] is None:
                    keys, values = xk_rotated, xv
                else:
                    keys = torch.cat([kv_cache[layer]['k'], xk_rotated], dim=1)
                    values = torch.cat([kv_cache[layer]['v'], xv], dim=1)
                
                kv_cache[layer] = {'k': keys, 'v': values}

                # --- 向量化 Attention (移除 Head 循环) ---
                # 目标形状: [batch, n_heads, seq_len, head_dim]
                # transpose: 交换 seq_len 和 n_heads 维度
                xq_heads = xq_rotated.transpose(1, 2) 
                keys_heads = keys.transpose(1, 2)
                values_heads = values.transpose(1, 2)

                # 处理 GQA (如果 n_kv_heads < n_heads)
                # 将 keys/values 在 head 维度复制 n_rep 倍，以匹配 queries
                if n_kv_heads < n_heads:
                    keys_heads = keys_heads.repeat_interleave(n_rep, dim=1)
                    values_heads = values_heads.repeat_interleave(n_rep, dim=1)
                
                # Attention Score: Q @ K.T
                # [1, n_heads, seq_len, head_dim] @ [1, n_heads, head_dim, total_seq_len] 
                # -> [1, n_heads, seq_len, total_seq_len]
                scores = torch.matmul(xq_heads, keys_heads.transpose(-2, -1)) / (head_dim ** 0.5)

                # Masking (仅在 Prefill 阶段需要，即 seq_len > 1)
                if seq_len > 1:
                    mask = torch.full((seq_len, seq_len), float("-inf"), device=DEVICE)
                    mask = torch.triu(mask, diagonal=1)
                    scores = scores + mask
                
                probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
                
                # Output: Probs @ V
                # [1, n_heads, seq_len, total_seq_len] @ [1, n_heads, total_seq_len, head_dim]
                # -> [1, n_heads, seq_len, head_dim]
                output = torch.matmul(probs, values_heads)
                
                # 恢复形状: [1, seq_len, dim]
                # 先 transpose 回 [1, seq_len, n_heads, head_dim] 然后 flatten
                output = output.transpose(1, 2).reshape(1, seq_len, dim)
                
                # Output Projection
                h_attn = torch.matmul(output, wo.T)
                h = h_residual + h_attn

                # --- FFN ---
                h_residual = h
                norm_ffn_w = model_weights[f"layers.{layer}.ffn_norm.weight"]
                h_norm = rms_norm(h, norm_ffn_w, norm_eps)
                
                w1 = model_weights[f"layers.{layer}.feed_forward.w1.weight"]
                w2 = model_weights[f"layers.{layer}.feed_forward.w2.weight"]
                w3 = model_weights[f"layers.{layer}.feed_forward.w3.weight"]
                
                # FFN 计算
                ffn_out = torch.matmul(
                    torch.nn.functional.silu(torch.matmul(h_norm, w1.T)) * torch.matmul(h_norm, w3.T), 
                    w2.T
                )
                h = h_residual + ffn_out

            # Final Output
            h_final = rms_norm(h, model_weights["norm.weight"], norm_eps)
            logits = torch.matmul(h_final[:, -1, :], model_weights["output.weight"].T)
            next_token = torch.argmax(logits, dim=-1)
            
            decoded = tokenizer.decode(next_token.item())
            print(decoded, end="", flush=True)
            
            # Update pointers
            curr_tokens = next_token.unsqueeze(0).unsqueeze(0)
            start_pos += seq_len
            
    print(f"\n\nGeneration speed: {max_new_tokens / (time.time() - t0):.2f} tokens/sec")

if __name__ == "__main__":
    main()