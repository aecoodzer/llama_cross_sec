import os
import torch
import threading
import struct
import time
import io
import json
import random

# --- 配置 ---
SERVER_HOST = '192.168.1.13'
SERVER_PORT = 11000


# --- 模型配置 ---
MODEL_PATH = "/workspace/code/modelscope/Llama-3-8B" # 请替换为你的实际路径
CONFIG_PATH = os.path.join(MODEL_PATH, "original/params.json")
MAX_SEQ_LEN = 8192  # 根据显存大小调整，通常 2048 或 4096
PROMPT_LEN = 4096
OUTPUT_TOKEN_NUM = 128

def rms_norm(tensor, norm_weights, norm_eps=1e-05):
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights

# 模型切分策略 (Llama3-8B 有 32 层)
# Client: [0, 1] ... [30, 31]
# Server: [2 ... 29]
CLOUD_START_LAYER = 2
CLOUD_END_LAYER = 30 

# 通信协议头: [Message Type (4 bytes int)] [Data Length (4 bytes int)]
# Msg Types
MSG_KV_CACHE = 1  # 发送 KV Cache
MSG_HIDDEN_REQ = 2 # 发送 Hidden State 请求计算
MSG_HIDDEN_RES = 3 # 返回计算后的 Hidden State

def get_prompt_len():
    return PROMPT_LEN

def get_device_edge():
    return "cuda:1" if torch.cuda.is_available() else "cpu"

def get_device_cloud():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def get_token_num():
    return OUTPUT_TOKEN_NUM

class LlamaConfig:
    def __init__(self):
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        self.dim = config["dim"]
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.n_kv_heads = config.get("n_kv_heads", self.n_heads)
        self.head_dim = self.dim // self.n_heads
        self.rope_theta = config.get("rope_theta", 500000.0)
        self.norm_eps = config["norm_eps"]

def send_packet(sock, msg_type, data_bytes, layer_idx=-1):
    """
    封装发送：Header + Payload
    Header: [Type(4), Length(4), LayerIdx(4)]
    """
    length = len(data_bytes)
    # pack: int, int, int
    header = struct.pack('!iii', msg_type, length, layer_idx)
    sock.sendall(header + data_bytes)

def recv_packet(sock):
    """
    封装接收：解析 Header 读取定长 Payload
    """
    header_size = 12 # 4+4+4
    header_data = _recv_n_bytes(sock, header_size)
    if not header_data:
        return None, None, None
    
    msg_type, length, layer_idx = struct.unpack('!iii', header_data)
    payload = _recv_n_bytes(sock, length)
    return msg_type, layer_idx, payload

def _recv_n_bytes(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def serialize_tensor(tensor_dict):
    """KV Cache 序列化: Dict -> Bytes"""
    buffer = io.BytesIO()
    torch.save(tensor_dict, buffer)
    return buffer.getvalue()

def deserialize_tensor(bytes_data, device):
    """反序列化: Bytes -> Dict"""
    buffer = io.BytesIO(bytes_data)
    return torch.load(buffer, map_location=device)

def deserialize_into_buffer(bytes_data, buffer_tensor):
    """
    [关键优化] 将序列化的 Tensor 直接加载到预分配的 buffer_tensor 中，避免新内存分配。
    注意：这要求 bytes_data 里的 Tensor 形状和 dtype 必须与 buffer_tensor 完全一致。
    """
    f = io.BytesIO(bytes_data)
    
    # torch.load 无法直接 load into，我们需要一个小技巧：
    # 先 load 到 CPU (这一步不可避免会产生 CPU 临时内存，但它是 pageable 的，开销比 CUDA malloc 小得多)
    # 然后 copy_ 到 GPU buffer
    
    # 也可以使用 pickle 的特定 hook，但最稳妥且改动最小的方式是：
    # 1. Load 到 CPU (临时对象)
    # 2. Async Copy 到 GPU Buffer
    # 3. 让临时对象立即销毁
    
    temp_tensor = torch.load(f, map_location="cpu")
    
    # 如果是字典 (KV Cache 情况)，需要特殊处理，这里主要优化 Hidden State (Tensor)
    if isinstance(temp_tensor, torch.Tensor):
        buffer_tensor.copy_(temp_tensor, non_blocking=True)
    elif isinstance(temp_tensor, dict) and 'h' in temp_tensor:
        # 处理 {'h': tensor, 'start_pos': int} 这种情况
        # 我们只把 tensor 部分 copy 进去
        buffer_tensor.copy_(temp_tensor['h'], non_blocking=True)
        return temp_tensor.get('start_pos', -1) # 返回 metadata
        
    return temp_tensor