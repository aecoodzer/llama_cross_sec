import torch
import transformers
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import os
import time

def rms_norm(tensor, norm_weights, norm_eps=1e-05):
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights

def main():
    # --- 1. 配置路径与加载模型 (保持不变) ---
    model_path = "/workspace/code/modelscope/Llama-3-8B"
    config_path = "/workspace/code/modelscope/Llama-3-8B/original"
    
    # 简单的路径检查，防止报错
    if not os.path.exists(model_path):
        print(f"Warning: Model path {model_path} not found. Please update paths.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    config_file = os.path.join(config_path, "params.json")
    with open(config_file, "r") as f:
        config = json.load(f)
        
    print("Loading model weights...")
    # 注意：这里假设你使用的是单文件pth，如果是分片模型需要合并加载
    model = torch.load(os.path.join(config_path, "consolidated.00.pth"), map_location="cpu")
    print("Model loaded.")

    # --- 2. 提取参数 ---
    dim = config["dim"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    n_kv_heads = config.get("n_kv_heads", n_heads) # 处理可能不存在的情况
    vocab_size = config["vocab_size"]
    norm_eps = config["norm_eps"]
    rope_theta = config.get("rope_theta", 500000.0) # Llama3 默认值
    
    # Llama 3 特有的 Grouped Query Attention (GQA) 分组倍数
    n_rep = n_heads // n_kv_heads 
    head_dim = dim // n_heads

    # --- 3. 准备 RoPE 频率 (预计算) ---
    # 预计算足够长的 RoPE table (例如 8192)
    max_seq_len = 2048
    freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, device="cpu", dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    # 转换为复数形式，方便旋转 (shape: [max_seq_len, head_dim // 2])
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    # --- 4. 准备推理输入 ---
    prompt = "The capital of China is"
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens).unsqueeze(0) # [1, seq_len]
    
    # --- KV Cache 初始化 ---
    # 列表索引对应层号，每层存储 {'k': tensor, 'v': tensor}
    kv_cache = [None] * n_layers

    # 生成设置
    max_new_tokens = 10
    generated_tokens = []
    
    print(f"\nPrompt: {prompt}")
    print("Generating:", end=" ", flush=True)

    # --- 5. 生成循环 (Generation Loop) ---
    start_pos = 0
    curr_tokens = tokens # 第一次循环是 Prompt (Prefill)，之后是单个 token (Decode)

    with torch.no_grad(): # 推理不需要梯度
        for _ in range(max_new_tokens):
            seq_len = curr_tokens.shape[1]
            
            # 5.1 Embedding
            embedding_layer = torch.nn.Embedding(vocab_size, dim)
            embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])
            # [1, seq_len, dim]
            h = embedding_layer(curr_tokens).to(torch.bfloat16)

            # 获取当前步的 RoPE 频率
            freqs_cis_curr = freqs_cis[start_pos : start_pos + seq_len]

            # 5.2 Layer Loop
            for layer in range(n_layers):
                # 保存残差
                h_residual = h
                # Pre-Normalization
                h_norm = rms_norm(h, model[f"layers.{layer}.attention_norm.weight"], norm_eps)

                # --- 加载权重 ---
                wq = model[f"layers.{layer}.attention.wq.weight"]
                wk = model[f"layers.{layer}.attention.wk.weight"]
                wv = model[f"layers.{layer}.attention.wv.weight"]
                wo = model[f"layers.{layer}.attention.wo.weight"]

                # --- Q, K, V 投影 ---
                # 为了效率，我们先 reshape 成 [batch, seq_len, n_heads, head_dim]
                # 注意：这里直接用线性层逻辑计算
                xq = torch.matmul(h_norm, wq.T).view(1, seq_len, n_heads, head_dim)
                xk = torch.matmul(h_norm, wk.T).view(1, seq_len, n_kv_heads, head_dim)
                xv = torch.matmul(h_norm, wv.T).view(1, seq_len, n_kv_heads, head_dim)

                # --- RoPE 旋转 ---
                # 将 Q 和 K 转换为复数进行旋转
                xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
                xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
                
                # 调整 freqs_cis 维度以匹配 [1, seq_len, 1, head_dim/2] 进行广播
                freqs_cis_curr_shaped = freqs_cis_curr.view(1, seq_len, 1, -1)
                
                xq_rotated = torch.view_as_real(xq_complex * freqs_cis_curr_shaped).flatten(3)
                xk_rotated = torch.view_as_real(xk_complex * freqs_cis_curr_shaped).flatten(3)
                
                # 转换回 bfloat16
                xq = xq_rotated.to(torch.bfloat16)
                xk = xk_rotated.to(torch.bfloat16)

                # --- KV Cache 更新机制 ---
                if kv_cache[layer] is None:
                    # Prefill 阶段：直接存储当前的 keys/values
                    keys = xk
                    values = xv
                else:
                    # Decode 阶段：拼接 历史KV + 当前KV
                    keys = torch.cat([kv_cache[layer]['k'], xk], dim=1)
                    values = torch.cat([kv_cache[layer]['v'], xv], dim=1)
                
                # 更新缓存
                kv_cache[layer] = {'k': keys, 'v': values}

                # --- Attention 计算 (GQA handled manually) ---
                # 输出容器
                output = torch.zeros_like(xq) # [1, seq_len, n_heads, head_dim]
                
                # 这里为了保留你的风格，我们手动处理 head，但使用了 Cache 后的 Keys/Values
                # keys shape: [1, total_seq_len, n_kv_heads, head_dim]
                
                for head in range(n_heads):
                    # 获取当前 head 的 Q [1, seq_len, head_dim]
                    q_head = xq[:, :, head, :]
                    
                    # 获取对应的 KV head (处理 GQA: n_heads 映射到 n_kv_heads)
                    kv_head_idx = head // n_rep
                    k_head = keys[:, :, kv_head_idx, :] # [1, total_seq_len, head_dim]
                    v_head = values[:, :, kv_head_idx, :]
                    
                    # Score: Q * K.T
                    # q_head: [1, curr_len, dim], k_head: [1, total_len, dim] -> [1, curr_len, total_len]
                    scores = torch.matmul(q_head, k_head.transpose(1, 2)) / (head_dim ** 0.5)
                    
                    # --- Masking ---
                    # 只有在 Prefill (start_pos == 0) 且 seq_len > 1 时才需要因果 Mask
                    if seq_len > 1:
                        mask = torch.full((seq_len, seq_len), float("-inf"), device=scores.device)
                        mask = torch.triu(mask, diagonal=1)
                        # 添加 Mask (注意：这只针对 Q 的长度部分，如果是 Decode 阶段 seq_len=1，无需 Mask)
                        scores = scores + mask
                    
                    probs = torch.nn.functional.softmax(scores, dim=-1).to(torch.bfloat16)
                    
                    # Output: Probs * V
                    head_out = torch.matmul(probs, v_head)
                    output[:, :, head, :] = head_out

                # Flatten heads: [1, seq_len, n_heads, head_dim] -> [1, seq_len, dim]
                attention_output = output.view(1, seq_len, dim)
                
                # Output Projection
                attention_output = torch.matmul(attention_output, wo.T)
                
                # Residual connection
                h = h_residual + attention_output

                # --- Feed Forward Network (FFN / MLP) ---
                h_residual = h
                h_norm = rms_norm(h, model[f"layers.{layer}.ffn_norm.weight"], norm_eps)
                
                w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
                w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
                w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
                
                # SwiGLU activation
                # (Silu(xW1) * xW3) W2
                ffn_out = torch.matmul(
                    torch.nn.functional.silu(torch.matmul(h_norm, w1.T)) * torch.matmul(h_norm, w3.T), 
                    w2.T
                )
                
                h = h_residual + ffn_out
            
            # --- 5.3 Final Prediction ---
            h_final = rms_norm(h, model["norm.weight"], norm_eps)
            # 我们只需要最后一个 token 的输出
            logits = torch.matmul(h_final[:, -1, :], model["output.weight"].T)
            
            # Greedy Decoding (Argmax)
            next_token = torch.argmax(logits, dim=-1)
            
            # 打印并存储
            decoded_token = tokenizer.decode(next_token.item())
            print(decoded_token, end="", flush=True)
            generated_tokens.append(next_token.item())
            
            # --- 6. 更新状态以进行下一步生成 ---
            curr_tokens = next_token.unsqueeze(0).unsqueeze(0) # [1, 1]
            start_pos += seq_len # 更新全局位置指针

    print("\n\nDone.")

if __name__ == "__main__":
    main()