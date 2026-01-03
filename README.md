# llama_cross_sec


用法：cross_sec文件夹为核心代码

* client.py 边缘端进程
* server.py 云端进程
* utils.py  通信协议和配置文件

## QUICK START
##### dependency：python 3.12/transformers/pytorch

##### remember adjust local model directory

```shell
cd llama_cross_sec
```

* cloud
```shell
python -m cross_sec.cloud
python -m pd_sl.cloud
```

* edge

```shell
python -m cross_sec.client
python -m pd_sl.client
```
##### 12.26 update
- 增加了泊松分布到达的请求队列发生器
- 测试了推理逻辑的正确性

##### 1.1 update
- 增加了简单的内存管理
- 实现了多线程的异步并发传输
- 实现了多线程的并发接收

to do：并发调度的kv cache池