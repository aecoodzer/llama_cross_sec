import torch
import transformers
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import matplotlib.pyplot as plt

import os
from safetensors.torch import load_file

def rms_norm(tensor, norm_weights, norm_eps=1e-05):
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights

def main():
    model_path = "/workspace/code/modelscope/Llama-3-8B"
    config_path = "/workspace/code/modelscope/Llama-3-8B/original"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.decode(tokenizer.encode("你好，世界！"))  # 测试分词器是否正常工作

    config_name = "params.json"
    config_path = os.path.join(config_path,config_name)
    with open(config_path, "r") as f:
        config = json.load(f)
    model = torch.load("/workspace/code/modelscope/Llama-3-8B/original/consolidated.00.pth")
    dim = config["dim"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    vocab_size = config["vocab_size"]
    multiple_of = config["multiple_of"]
    ffn_dim_multiplier = config["ffn_dim_multiplier"]
    norm_eps = config["norm_eps"]
    rope_theta = torch.tensor(config["rope_theta"])
    prompt = "the capital of china is"
    tokens =tokenizer.encode(prompt)
    # print(tokens)
    tokens = torch.tensor(tokens)
    prompt_split_as_tokens = [tokenizer.decode([token.item()]) for token in tokens]
    length_of_tokens = len(prompt_split_as_tokens)

    embedding_layer = torch.nn.Embedding(vocab_size, dim)
    embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])
    token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)

    zero_to_one_split_into_64_parts = torch.tensor(range(64))/64

    freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts)
    freqs_for_each_token = torch.outer(torch.arange(length_of_tokens), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)

    final_embedding = token_embeddings_unnormalized
    for layer in range(n_layers):
        qkv_attention_store = []
        layer_embedding_norm = rms_norm(final_embedding, model[f"layers.{layer}.attention_norm.weight"])
        q_layer = model[f"layers.{layer}.attention.wq.weight"]
        q_layer = q_layer.view(n_heads, q_layer.shape[0] // n_heads, dim)
        k_layer = model[f"layers.{layer}.attention.wk.weight"]
        k_layer = k_layer.view(n_kv_heads, k_layer.shape[0] // n_kv_heads, dim)
        v_layer = model[f"layers.{layer}.attention.wv.weight"]
        v_layer = v_layer.view(n_kv_heads, v_layer.shape[0] // n_kv_heads, dim)
        w_layer = model[f"layers.{layer}.attention.wo.weight"]
        for head in range(n_heads):
            q_layer_head = q_layer[head]
            k_layer_head = k_layer[head//4]
            v_layer_head = v_layer[head//4]
            q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
            k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
            v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)
            q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
            q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
            q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis)
            q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)
            k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
            k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
            k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
            k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
            qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(128)**0.5
            mask = torch.full((len(token_embeddings_unnormalized), len(token_embeddings_unnormalized)), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            qk_per_token_after_masking = qk_per_token + mask
            qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
            qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
            qkv_attention_store.append(qkv_attention)
    
        stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
        w_layer = model[f"layers.{layer}.attention.wo.weight"]
        embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
        embedding_after_edit = final_embedding + embedding_delta
        embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[f"layers.{layer}.ffn_norm.weight"])
        w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
        w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
        w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
        output_after_feedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
        final_embedding = embedding_after_edit+output_after_feedforward

    final_embedding = rms_norm(final_embedding, model["norm.weight"])
    logits = torch.matmul(final_embedding[-1], model["output.weight"].T)
    next_token = torch.argmax(logits, dim=-1)
    print(tokenizer.decode([next_token.item()]))

if __name__ == "__main__":
    main()