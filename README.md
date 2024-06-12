## 安装基础文件
```{bash}
##更新基础库
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && sudo apt-get install git-lfs && git lfs install

#安装依赖环境
pip install -r requirements.txt

#安装jdk
#1. 更新apt库
sudo apt update

#2. 安装jdk
apt install openjdk-21-jdk
```

## 下载模型
```{bash}
bash ./Model/models.sh

```

## 训练分类器
### 准备数据

```
## 训练数据
./Intention_classify/data/Insurance/train_list_data.jsonl
## 测试数据
./Intention_classify/data/Insurance/train_list_data.jsonl

## 数据格式为:
[
{"sentence": "糖尿病，高血压既往疾病的客户是否可以投保，一般情况下核保结论是？", "label": 1},
 {"sentence": "你的解释我不太理解，能让我和真人客服通话吗？", "label": 2}
]
```

### 训练分类器
```{bash}
cd Intention_classify
python train.py
```

## 训练embedding模型(可选)
```
bash embedding_sft_hn.sh
bash embedding_sft.sh

## 训练数据
./data/embedding_sft.jsonl
## 数据格式为:
{"query": "如何能获取到最新的MSH保险产品的保险方案手册、销售宣传册和投保表格等文档？", "pos": ["在哪里可以找到MSH相关保险产品最新的保险计划书、销售单页、投保单等材料？"], "neg": ["经典、精选和欣享人生医疗保险产品费率有效期多久？每年费率会有变化吗？"]}
{"query": "在何处能够找到关于MSH保险的最新保单样本、产品简介页和申请表格等资料？", "pos": ["在哪里可以找到MSH相关保险产品最新的保险计划书、销售单页、投保单等材料？"], "neg": ["经典、精选和欣享人生医疗保险产品费率有效期多久？每年费率会有变化吗？"]}
```

## 配置文件
```{Python}
#修改文件./RAG_pipeline/QA_math.py的
    dashscope.api_key
#修改文件./RAG_pipeline/QA_math.py的
    dashscope.api_key

#修改文件./user.py的
    tokenizer_model_path
    classify_model_path
```