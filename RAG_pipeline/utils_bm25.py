import json
import logging
import os
from datetime import datetime
from .src.retriever import DenseRetriever, SparseRetriever, Reranker
from .conf.conf import load_config

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

def creat_retriever(type):
    app_conf = load_config(type)
    logger = init_logger(log_dir=app_conf['log_dir'])
    retriever = SparseRetriever(model_name_or_path='bm25',
                                index_dir=app_conf['corpus']['index_dir'],
                                language=app_conf['corpus']['language'],
                                threads=app_conf['retrieve_model']['threads'],
                                batch_size=app_conf['retrieve_model']['batch_size'])
    return retriever


