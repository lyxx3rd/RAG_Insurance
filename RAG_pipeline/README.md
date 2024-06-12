# RAG-pipeline

## 安装环境

安装必要python环境
```
pip install -r requirements.txt
```

安装必要java环境
```
## 更新apt
sudo apt update
## 安装openjdk-21-jdk
apt install openjdk-21-jdk
```

## BM25 retrivever

1. 生成索引
```
bash Indexing.sh
```

2. 修改配置文件

修改 ./src/retriever/server/conf/config.yaml 配置文件
```
log_dir:

retrieve_model:
  model_name_or_path: "bm25"

corpus:
  language: zn
  index_dir: ./indexes/data_to_encode/
```

2.  启用服务
```
python server.py
```