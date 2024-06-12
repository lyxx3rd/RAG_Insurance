import sys
from io import StringIO
import json
from .Indexing import check_indexing
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher, AutoQueryEncoder
from pyserini.search.hybrid import HybridSearcher
from FlagEmbedding import FlagReranker

def lunch_process():
    check_indexing()
    ssearcher = LuceneSearcher('./indexes/Insurance/bm25_index/Contract_index')
    encoder = AutoQueryEncoder('./Model/bge-m3')
    dsearcher = FaissSearcher(index_dir = "./indexes/Insurance/dense_index/Contract_index",query_encoder = encoder)
    hsearcher = HybridSearcher(dsearcher, ssearcher)
    reranker = FlagReranker('./Model/bge-reranker-base', use_fp16=True)
    
    contract_dict = []
    with open("./RAG_data/Contract_jsonl/Contract_list.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            # 每一行都是一个JSON字符串，使用json.loads转换为字典
            contract_dict.append(json.loads(line.strip()))
    qa_dict = []
    with open("./RAG_data/QA_jsonl/QA_list.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            # 每一行都是一个JSON字符串，使用json.loads转换为字典
            qa_dict.append(json.loads(line.strip()))
            
    with open("./data/QA_dict.json", 'r', encoding='utf-8') as f:
        QA_answer_dict = json.load(f)

    ssearcher_QA = LuceneSearcher('./indexes/Insurance/bm25_index/QA_index')
    encoder_QA = AutoQueryEncoder('./Model/mixed_model_1')
    dsearcher_QA = FaissSearcher(index_dir = "./indexes/Insurance/dense_index/QA_index",query_encoder = encoder_QA)
    hsearcher_QA = HybridSearcher(dsearcher_QA, ssearcher_QA)

    return hsearcher,contract_dict,qa_dict,QA_answer_dict,reranker,hsearcher_QA
