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
from utils import Call_with_messages,DatasetClassify

print("欢迎使用范云阳号保险AI模型，模型加载中")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-5

word2id = {"闲聊":0,"咨询":1,"转人工":2}
id2word = {0:"闲聊",1:"咨询",2:"转人工"}

def user_chat(classify_model_path="./Model_save/classify_model.pt", tokenizer_model_path = "./Model/chinese-roberta-wwm-ext"):
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
    while True:
        question = input("请输入一些内容：")
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
            print("启动Qwen_范云阳号聊天程序")
            conversation_history.append({'role': 'user', 'content': question})
            # 调用chat_with_qwen函数，传入当前的对话历史
            response_content = Call_with_messages(conversation_history)
            print("范云阳：", response_content)
            # 更新对话历史，准备下一轮对话
            conversation_history.append({'role': 'assistant', 'content': response_content})
        if Intention == 1:
            print("启动Qwen_范云阳号QA匹配及合同释义程序\n")
            print("程序开发中\n")
        if Intention == 2:
            print("启动Qwen_范云阳号人工聊天程序\n")
            print("程序开发中\n")