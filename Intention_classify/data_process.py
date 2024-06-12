word2id = {"闲聊":0,"咨询":1,"转人工":2}
id2word = {0:"闲聊",1:"咨询",2:"转人工"}

# 打开文件，'r' 表示以只读模式打开
with open('./data/Insurance/QA.txt', 'r', encoding='utf-8') as file:
    # 使用 read() 方法读取文件的全部内容
    lines = file.readlines()
content = [line.strip() for line in lines]

data_list = []
for i in range(len(content)):
    dict_temp = {}
    dict_temp["sentence"] = content[i]
    dict_temp["label"] = "咨询"
    data_list.append(dict_temp)

## 闲聊
with open('./data/Insurance/chat.txt', 'r', encoding='utf-8') as file:
    # 使用 read() 方法读取文件的全部内容
    lines = file.readlines()
content = [line.strip() for line in lines]

for i in range(len(content)):
    dict_temp = {}
    dict_temp["sentence"] = content[i]
    dict_temp["label"] = "闲聊"
    data_list.append(dict_temp)

## 人工
with open('./data/Insurance/human.txt', 'r', encoding='utf-8') as file:
    # 使用 read() 方法读取文件的全部内容
    lines = file.readlines()
content = [line.strip() for line in lines]

for i in range(len(content)):
    dict_temp = {}
    dict_temp["sentence"] = content[i]
    dict_temp["label"] = "转人工"
    data_list.append(dict_temp)

data_id_list = data_list
for i in range(len(data_list)):
    data_id_list[i]["label"] = word2id[data_list[i]["label"]]

import random

def split_list_by_ratio(lst, ratio=0.2):
    """
    将列表lst按照指定比例ratio随机切分成两个子列表。
    
    参数:
    lst -- 要切分的列表
    ratio -- 切分比例，例如0.2表示20%的数据放在一个列表，剩余80%放在另一个列表
    
    返回:
    两个子列表
    """
    # 复制列表以避免修改原列表
    shuffled_lst = lst.copy()
    random.shuffle(shuffled_lst)
    
    # 计算分割点
    split_point = int(len(shuffled_lst) * ratio)
    
    # 切分列表
    sublist1 = shuffled_lst[:split_point]
    sublist2 = shuffled_lst[split_point:]
    
    return sublist1, sublist2

# 示例使用
train_data, test_data = split_list_by_ratio(data_id_list, 0.8)

import json
with open('./data/Insurance/train_list_data.jsonl', 'w', encoding='utf-8') as file:
    # 使用 read() 方法读取文件的全部内容
    json.dump(train_data, file, ensure_ascii=False)
with open('./data/Insurance/test_list_data.jsonl', 'w', encoding='utf-8') as file:
    # 使用 read() 方法读取文件的全部内容
    json.dump(test_data, file, ensure_ascii=False)
