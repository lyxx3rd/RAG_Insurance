"""
本文件是用来生成分类数据的。
每一个query，生成两个例子，一个是正例，另一个是负例。负例从接口中调取。
"""
import numpy as np
import json
import os.path
import random
from dataclasses import dataclass, asdict
from typing import Optional

from loguru import logger
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput

logger.add("out.log")
import numpy
import requests
import torch.cuda
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, AutoTokenizer, AutoModel
from sklearn.metrics import classification_report

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-5

model_path = "../Model/chinese-roberta-wwm-ext"

class DatasetClassify(Dataset):
    def __init__(self, path):
        self.data_list = json.load(open(path, encoding='utf8'))

    def __getitem__(self, index):
        item = self.data_list[index]
        item = DataItem(**item)
        content = item.sentence
        return content, item.label

    def __len__(self):
        return len(self.data_list)


def collator_fn(batch):
    batch = numpy.array(batch)

    data_batch = batch[:, 0]
    label_batch = numpy.array(batch[:, 1], dtype=int)
    data_batch = tokenizer(data_batch.tolist(), max_length=256, padding=True, truncation=True,
                           return_tensors="pt").to(DEVICE)
    return data_batch, torch.tensor(label_batch, device=DEVICE, dtype=torch.long)


@dataclass
class DataItem:
    sentence: str
    label: int


@torch.no_grad()
def eval():
    num_true = 0
    num_total = 0
    for item, label in tqdm(dev_data_loader, position=0, leave=True):
        output = model(**item, labels=label)
        pre_label = output.logits.detach().cpu().numpy()
        real_label = label.detach().cpu().numpy()
        pre_label = np.argmax(pre_label, axis=1)
        num_true += np.sum(real_label == pre_label)
        num_total += len(pre_label)
    acc = num_true/num_total
    logger.info("\n" + str(acc))
    return acc


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path,num_labels=3)
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
train_data_loader = DataLoader(DatasetClassify("./data/Insurance/train_list_data.jsonl"), batch_size=64, shuffle=True,
                               collate_fn=collator_fn)
dev_data_loader = DataLoader(DatasetClassify("./data/Insurance/test_list_data.jsonl"), batch_size=64, shuffle=False,
                             collate_fn=collator_fn)
EPOCHS = 10
step = 0
accu_max = 0.0
num_training_steps = len(train_data_loader) * EPOCHS
for epoch in range(EPOCHS):
    loss_total = 0.0
    for index, (item, label) in enumerate(tqdm(train_data_loader), start=1):
        step = epoch * len(train_data_loader) + index
        output = model(labels=label, **item)
        loss = output.loss
        loss_total += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.info(f"第{step}步的损失为{loss_total}")
        loss_total = 0.0
        model.eval()
        accu_score = eval()
        model.train()
        if accu_score > accu_max:
            accu_max = accu_score
            torch.save(model, f"./Model_save/classify_model.pt")
        if epoch > 0:
            LR = LR * 0.6


        
