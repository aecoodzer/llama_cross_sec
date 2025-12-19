# llama_cross_sec

本代码使用于科研用途。

用法：cross_sec文件夹为核心代码

* client.py 边缘端进程
* server.py 云端进程
* utils.py  通信协议和配置文件

## QUICK START
##### dependency：python 3.12/transformers/pytorch

##### remember adjust local model directory

```shell
cd cross_sec
```

* cloud
```shell
python server.py
```

* edge

```shell
python client.py
```
