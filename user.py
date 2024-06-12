import numpy as np
import json
import os.path
from loguru import logger
import numpy
import torch.cuda
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import classification_report
from Intention_classify.utils import Call_with_messages,DatasetClassify
from RAG_pipeline.QA_math import QA_math_Qwen,QA_math_rerank
from RAG_pipeline.Contract_search import Qwen_search
from RAG_pipeline.lunch_process import lunch_process


print("欢迎使用范云阳号保险AI模型，模型加载中")
tokenizer_model_path = "./Model/chinese-roberta-wwm-ext"
classify_model_path="./Model_save/classify_model.pt"
hsearcher,contract_dict,qa_dict,QA_answer_dict,reranker,hsearcher_QA = lunch_process()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-5

word2id = {"闲聊":0,"咨询":1,"转人工":2}
id2word = {0:"闲聊",1:"咨询",2:"转人工"}

def Qwen_and_QA(QA_answer_dict,qa_dict,contract_dict, question,reranker, usage_rerank = False,hsearcher_QA=None):
    if usage_rerank == True:
        ide, q, a = QA_math_rerank(QA_answer_dict,qa_dict, question, reranker,hsearcher_QA)
    else:
        ide, q, a = QA_math_Qwen(QA_answer_dict,qa_dict, question,hsearcher_QA)
    if ide == 0:
        print("【根据您输入的问题,您可以参考一下问题以及回答】")
        print("【参考问题】:",q)
        print("【参考回答】:",a)
        question = ""
    if ide == 1:
        print("正在搜索相关条款，请稍后...")
        res, item1, item2, item3 = Qwen_search(question,hsearcher,contract_dict)
        print("【您咨询的问题是】：",question)
        print("【范云阳】：",res)
        question = input("如需显示具体条款请输入【显示条款】,或直接输入新问题:")
        if question == "显示条款":
            print(item1,"\n")
            print(item2,"\n")
            print(item3,"\n")
            question == ""
            question = input("【系统】：您还可以输入其他的咨询问题:")
    return question

def user_chat(classify_model_path="./Intention_classify/Model_save/classify_model.pt", tokenizer_model_path = "./Model/chinese-roberta-wwm-ext"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
    model = torch.load(classify_model_path)

    def collator_fn(batch):
        batch = np.array(batch)
    
        data_batch = batch[:, 0]
        label_batch = np.array(batch[:, 1], dtype=int)
        data_batch = tokenizer(data_batch.tolist(), max_length=256, padding=True, truncation=True,
                               return_tensors="pt").to(DEVICE)
        return data_batch, torch.tensor(label_batch, device=DEVICE, dtype=torch.long)
    
    print("模型加载成功，请输入内容开始对话，输入“end”结束对话")
    conversation_history = [{'role': 'system', 'content': '你是Qwen_范云阳号聊天程序'}]
    question = ""
    while True:
        if question == "":
            question = input("请输入您的问题:")
        if question.lower() == "end":
            break
        model.eval()
        pred_label = []
        test_data_loader = DataLoader(DatasetClassify(question), batch_size=32, shuffle=False,
                                     collate_fn=collator_fn)
        for item, label in test_data_loader:
            model.eval()
            output = model(**item)
            pre_label = output.logits.detach().cpu().numpy()
            pre_label = np.argmax(pre_label, axis=1)
        Intention = int(pre_label[0])
        if Intention == 0:
            conversation_history.append({'role': 'user', 'content': question})
            # 调用chat_with_qwen函数，传入当前的对话历史
            response_content = Call_with_messages(conversation_history)
            print("范云阳：", response_content)
            # 更新对话历史，准备下一轮对话
            conversation_history.append({'role': 'assistant', 'content': response_content})
            question = ""
        if Intention == 1:
            question = Qwen_and_QA(QA_answer_dict,qa_dict,contract_dict, question,reranker,usage_rerank=False,hsearcher_QA=hsearcher_QA)
        if Intention == 2:
            print("程序开发中\n")
            question = ""
            
user_chat()