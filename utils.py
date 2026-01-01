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


class InferenceRequest:
    def __init__(self, req_id, prompt_len, output_len):
        self.req_id = req_id
        self.arrival_time = time.time() # 记录到达时间（进入队列的时间）
        self.prompt_len = prompt_len
        self.output_len = output_len
        # 模拟随机 Input Token (为了性能测试，直接生成随机Tensor比Tokenizer更快)
        # 词表大小假设为 32000 (Llama 标准)
        self.input_tokens = torch.randint(0, 32000, (1, prompt_len), dtype=torch.long)

class WorkloadGenerator(threading.Thread):
    def __init__(self, request_queue, arrival_rate, max_requests):
        super().__init__()
        self.request_queue = request_queue
        self.arrival_rate = arrival_rate # Lambda (requests/second)
        self.max_requests = max_requests
        self.stop_event = threading.Event()

    def run(self):
        print(f"[Generator] Started. Target Rate: {self.arrival_rate} req/s")
        for i in range(self.max_requests):
            if self.stop_event.is_set():
                break
            
            # 1. 生成请求
            req = InferenceRequest(req_id=i, prompt_len=4096, output_len=128)
            self.request_queue.put(req)
            print(f"[Generator] Request {i} arrived. Queue size: {self.request_queue.qsize()}")
            
            # 2. 泊松分布等待
            # random.expovariate(lambda) 返回符合指数分布的时间间隔
            # 这是泊松过程的标准模拟方法
            if self.arrival_rate > 0:
                sleep_time = random.expovariate(self.arrival_rate)
                time.sleep(sleep_time)
        
        print("[Generator] All requests generated.")