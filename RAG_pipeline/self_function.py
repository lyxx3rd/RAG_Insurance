import os
import shutil
import json

def prepare_directory(directory_path):
    # 检查目录是否存在
    if not os.path.exists(directory_path):
        # 如果不存在，则创建目录
        os.makedirs(directory_path)
        print(f"Directory {directory_path} created.")
    else:
        # 如果目录存在，则清空目录
        # 注意：这将删除目录下的所有文件和子目录
        shutil.rmtree(directory_path)
        os.makedirs(directory_path)
        print(f"Directory {directory_path} cleared.")


def check_file_num(directory_path):
# 指定要列出文件的目录
    # 使用os.listdir()获取目录内容列表
    contents = os.listdir(directory_path)
    
    # 过滤出文件（排除子目录）
    files_only = [item for item in contents if os.path.isfile(os.path.join(directory_path, item))]
    
    # 返回文件数量以及名称
    return len(files_only), files_only

def modification_time(directory_path):
    num_file, file_list=check_file_num(directory_path)
    if num_file == 0:
        print(f"{directory_path}文件夹无数据")
        sys.exit(0)
    elif num_file > 1:
        print(f"{directory_path}文件夹文件数量大于1，暂时仅支持单一文件处理")
        sys.exit(0)
    else:
        file_name = directory_path +"/"+ str(file_list[0])
        parent_dir_path = os.path.dirname(file_name)
        # 获取上级目录的名称
        parent_dir_name = os.path.basename(parent_dir_path)
        modification_time = os.path.getmtime(file_name)
    return parent_dir_name, int(modification_time)

def check_dict_and_modify_time(RAG_data_modification_time_path = "./conf/RAG_data_modification_time.json",path_list = ["./RAG_data/QA_json","./RAG_data/QA_jsonl","./RAG_data/Contract_json","./RAG_data/Contract_jsonl"]):
    if not os.path.isfile(RAG_data_modification_time_path):
        dict = {}
        for input_path in path_list:
            file_name, time = modification_time(input_path)
            dict[file_name] = time
        with open("./conf/RAG_data_modification_time.json", 'w', encoding='utf-8') as f:
            json.dump(dict,f)
        ## 无历史记录，需要更新
        print("目录更新时间缺失，开始更新目录")
        return 1
    with open("./conf/RAG_data_modification_time.json", 'r', encoding='utf-8') as f:
        dict = json.load(f)
        ## 有历史记录
    i = 0
    for directory_path in path_list:
        file_name, time = modification_time(directory_path)
        try:
            if dict[file_name] != time:
                i = i+1
                ## 时间不相等，原始数据存在变化，需要更新Index和dict
                dict[file_name] = time
                with open("./conf/RAG_data_modification_time.json", 'w', encoding='utf-8') as f:
                    json.dump(dict,f)
        except KeyError:
            print("存在新的数据文件夹或文件夹路径设置错误")
            i = i+1
            dict[file_name] = time
            with open("./conf/RAG_data_modification_time.json", 'w', encoding='utf-8') as f:
                json.dump(dict,f)
    if i>0:
        print("一共",i,"个数据发生变化，开始更新目录")
        return 1
    else:
        return 0
                
 
