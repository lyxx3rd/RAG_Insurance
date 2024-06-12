import json
import logging
import os
from datetime import datetime
from src.retriever import DenseRetriever, SparseRetriever, Reranker
from .conf.conf import app_conf


def init_logger(log_dir: str):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_name = f'server_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'
    fhl = logging.FileHandler(filename=os.path.join(log_dir, file_name), encoding='utf-8', mode='w')
    fhl.setLevel(logging.INFO)
    fmt = '%(asctime)s - [%(process)d] - %(levelname)s - %(filename)s - %(lineno)d --- %(message)s'
    date_fmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt, date_fmt)
    fhl.setFormatter(formatter)
    logger.addHandler(fhl)
    return logger


logger = init_logger(log_dir=app_conf['log_dir'])

if app_conf['retrieve_model']['model_name_or_path'] == 'bm25':
    retriever = SparseRetriever(model_name_or_path='bm25',
                                index_dir=app_conf['corpus']['index_dir'],
                                language=app_conf['corpus']['language'],
                                threads=app_conf['retrieve_model']['threads'],
                                batch_size=app_conf['retrieve_model']['batch_size'])
else:
    retriever = DenseRetriever(model_name_or_path=app_conf['retrieve_model']['model_name_or_path'],
                               index_dir=app_conf['corpus']['index_dir'],
                               device=app_conf['retrieve_model']['device'],
                               pooling=app_conf['retrieve_model']['pooling'],
                               l2_norm=app_conf['retrieve_model']['l2_norm'],
                               corpus_path=app_conf['corpus']['corpus_path'],
                               query_max_length=app_conf['query']['max_length'],
                               query_prefix=app_conf['query']['prefix'],
                               threads=app_conf['retrieve_model']['threads'],
                               batch_size=app_conf['retrieve_model']['batch_size'],
                               ef_search=app_conf['retrieve_model']['ef_search'])

if app_conf['reranker']['model_name_or_path']:
    reranker = Reranker(model_name_or_path=app_conf['reranker']['model_name_or_path'],
                        batch_size=app_conf['reranker']['batch_size'],
                        cutoff_layers=app_conf['reranker']['cutoff_layers'],
                        device=app_conf['reranker']['device'])
else:
    reranker = None

k = app_conf['retrieve_model']['k']
query_to_pos_path = app_conf['query']['query_to_pos_path']
query_to_pos = {}
if query_to_pos_path is not None:
    with open(query_to_pos_path, 'r') as f:
        lines = [json.loads(line) for line in f.readlines()]
    for line in lines:
        query_to_pos[line['query']] = {str(pos['docid']) for pos in line['pos']}