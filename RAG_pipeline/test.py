import subprocess
import json
from Contract_search import Qwen_search
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher, AutoQueryEncoder
from pyserini.search.hybrid import HybridSearcher

print("开始加载程序")
ssearcher = LuceneSearcher('./indexes/Insurance/bm25_index/Contract_index')
encoder = AutoQueryEncoder('../Model/bge-m3')
dsearcher = FaissSearcher(index_dir = "./indexes/Insurance/dense_index/Contract_index",query_encoder = encoder)
hsearcher = HybridSearcher(dsearcher, ssearcher)

contract_dict = []
with open("./RAG_data/Contract_jsonl/Contract_list.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        # 每一行都是一个JSON字符串，使用json.loads转换为字典
        contract_dict.append(json.loads(line.strip()))

res = Qwen_search("如果我接下来要前往欧洲国家外派工作半年的时间，是否可以购买你们的经典个人产品吗？",hsearcher,contract_dict)
print(res)
