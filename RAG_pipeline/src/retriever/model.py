import copy
import json

import numpy as np
from typing import Union, Any, Dict, List

from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher, AutoQueryEncoder, DprQueryEncoder
from pyserini.encode import AutoDocumentEncoder
from FlagEmbedding import LayerWiseFlagLLMReranker, FlagReranker


class Retriever:
    def __init__(self,
                 model_name_or_path: str,
                 index_dir: str,
                 batch_size: int,
                 threads: int,
                 model_type: str):
        self.model_name_or_path = model_name_or_path
        self.index_dir = index_dir
        self.batch_size = batch_size
        self.threads = threads
        self.model_type = model_type
        self.searcher = None

    def init_searcher(self):
        raise NotImplementedError

    def get_doc(self, docid: str):
        raise NotImplementedError

    def retrieve(self,
                 queries: list[str],
                 query_ids: list[str] = None,
                 k: int = 10
                 ) -> list[list[dict[str, Any]]]:
        if query_ids is None:
            query_ids = [str(idx) for idx in range(len(queries))]
        assert len(queries) == len(query_ids)
        all_results = []
        for index in range(0, len(queries), self.batch_size):
            batch_queries = queries[index: index + self.batch_size]
            batch_query_ids = query_ids[index: index + self.batch_size]
            threads = min(self.threads, len(batch_queries))
            if self.model_type == 'sparse':
                batch_hits = self.searcher.batch_search(queries=batch_queries, qids=batch_query_ids, k=k, threads=threads)
            else:
                batch_hits = self.searcher.batch_search(queries=batch_queries, q_ids=batch_query_ids, k=k, threads=threads)
            batch_results = []
            for query_id in batch_query_ids:
                result = []
                for rank, hit in enumerate(batch_hits[query_id], start=1):
                    result.append({
                        'query_id': query_id,
                        'rank': rank,
                        'score': round(float(hit.score), 3),
                        **self.get_doc(hit.docid)
                    })
                batch_results.append(result)
            all_results.extend(batch_results)
        return all_results


class SparseRetriever(Retriever):
    def __init__(self,
                 model_name_or_path: str,
                 index_dir: str,
                 language: str,
                 threads: int = 1,
                 batch_size: int = 32):
        super().__init__(model_name_or_path, index_dir, batch_size, threads, model_type='sparse')
        self.searcher = self.init_searcher(language)

    def init_searcher(self, language='en'):
        searcher = LuceneSearcher(index_dir=self.index_dir)
        if self.model_name_or_path == 'bm25':
            searcher.set_bm25(0.82, 0.68)
        else:
            raise ValueError(f'Invalid model_name_or_path: "{self.model_name_or_path}"')
        searcher.set_language(language)
        return searcher

    def get_doc(self, docid: str) -> Dict[str, Any]:
        doc = json.loads(self.searcher.doc(docid=docid).raw())
        return {'docid': docid, 'text': doc['contents']}


class DenseRetriever(Retriever):
    def __init__(self, model_name_or_path: str,
                 index_dir: str,
                 batch_size: int,
                 threads: int,
                 device: str,
                 pooling: str,
                 l2_norm: bool,
                 corpus_path: str,
                 query_max_length: int = 512,
                 query_prefix: str = None,
                 ef_search: int = None):
        super().__init__(model_name_or_path, index_dir, batch_size, threads, model_type='dense')
        self.device = device
        self.query_prefix = query_prefix
        self.corpus = self.init_corpus(corpus_path)
        self.searcher = self.init_searcher(pooling, l2_norm, ef_search, query_max_length, query_prefix)
        self.document_encoder = AutoDocumentEncoder(model_name=model_name_or_path,
                                                    tokenizer_name=model_name_or_path,
                                                    device=device,
                                                    pooling=pooling,
                                                    l2_norm=l2_norm)

    @staticmethod
    def init_corpus(corpus_path: str) -> dict[Any, dict[str, Any]]:
        corpus = {}
        if corpus_path.endswith('.tsv'):
            with open(corpus_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip('\n').split('\t')
                    if len(line) == 2:
                        corpus[str(line[0])] = {'docid': line[0], 'text': line[1]}
                    else:
                        corpus[str(line[0])] = {'docid': line[0], 'text': line[1], 'title': line[2]}
        elif corpus_path.endswith('.jsonl'):
            with open(corpus_path, 'r') as f:
                for line in f.readlines():
                    line = json.loads(line)
                    if 'docid' in line.keys():
                        docid = line['docid']
                    elif 'id' in line.keys():
                        docid = line['id']
                        line['docid'] = docid
                        del line['id']
                    elif '_id' in line.keys():
                        docid = line['_id']
                        line['docid'] = docid
                        del line['_id']
                    else:
                        raise KeyError(f'Missing key "docid", "id" or "_id" in file "{corpus_path}"')
                    if 'contents' in line.keys():
                        line['text'] = line['contents']
                        del line['contents']
                    corpus[str(docid)] = line
        else:
            raise ValueError(f"Unsupported data format: \"{corpus_path}\"")
        return corpus

    def init_searcher(self, pooling='cls', l2_norm=None, ef_search=None, query_max_length=512, query_prefix=None):
        if 'dpr' in self.model_name_or_path.lower():
            encoder = DprQueryEncoder(encoder_dir=self.model_name_or_path, pooling=pooling, l2_norm=l2_norm,
                                      device=self.device, max_length=query_max_length, prefix=query_prefix)
        else:
            encoder = AutoQueryEncoder(encoder_dir=self.model_name_or_path, pooling=pooling, l2_norm=l2_norm,
                                       device=self.device, max_length=query_max_length, prefix=query_prefix)
        searcher = FaissSearcher(
            index_dir=self.index_dir,
            query_encoder=encoder
        )
        if ef_search:
            searcher.set_hnsw_ef_search(ef_search)
        return searcher

    def get_doc(self, docid: str) -> Dict[str, Any]:
        return self.corpus[docid]

    def vectorize(self, texts: list[str], is_query: bool = True) -> np.ndarray:
        if is_query:
            embeddings = np.array([self.searcher.query_encoder.encode(text) for text in texts])
        else:
            embeddings = []
            for index in range(0, len(texts), self.batch_size):
                batch_texts = texts[index: index + self.batch_size]
                embeddings.append(np.array(self.document_encoder.encode(batch_texts)))
            embeddings = np.concatenate(embeddings, axis=0)
        n, m = embeddings.shape
        assert n == len(texts) and m == self.searcher.dimension
        return embeddings

    def mix_retrieve(self,
                     queries1: Union[list[str], np.ndarray],
                     queries2: Union[list[str], np.ndarray],
                     query_ids: list[str] = None,
                     mix_ratio: float = 0.5, k: int = 10
                     ) -> list[list[dict[str, Union[str, int, float]]]]:
        assert 0 < len(queries1) == len(queries2)
        if isinstance(queries1[0], str):
            queries1 = self.vectorize(queries1)
        if isinstance(queries2[0], str):
            queries2 = self.vectorize(queries2)
        mixed_vectors = mix_ratio * queries1 + (1 - mix_ratio) * queries2
        return self.retrieve(queries=mixed_vectors, query_ids=query_ids, k=k)

    def rerank(self,
               queries: List[str],
               documents: List[List[Dict[str, Any]]],
               query_ids: List[str] = None
               ) -> List[List[Dict[str, Any]]]:
        assert len(documents) == len(queries)
        if query_ids is None:
            query_ids = [str(idx) for idx in range(len(queries))]
        assert len(queries) == len(query_ids)
        all_results = []
        for index in range(0, len(queries), self.batch_size):
            batch_documents = documents[index: index + self.batch_size]
            batch_queries = queries[index: index + self.batch_size]
            batch_query_ids = query_ids[index: index + self.batch_size]
            batch_doc_vectors = []
            for result in batch_documents:
                mini_batch_doc_vectors = []
                for mini_index in range(0, len(result), self.batch_size):
                    mini_batch_documents = result[mini_index: mini_index + self.batch_size]
                    mini_batch_doc_vectors.extend(self.vectorize([d['text'] for d in mini_batch_documents],
                                                                 is_query=False))
                batch_doc_vectors.append(mini_batch_doc_vectors)
            if isinstance(batch_queries[0], str):
                batch_query_vectors = self.vectorize(batch_queries, is_query=True)
            else:
                batch_query_vectors = batch_queries
            batch_results = []
            for query_id, query_vector, documents, doc_vectors in zip(batch_query_ids, batch_query_vectors,
                                                                      batch_documents, batch_doc_vectors):
                result = []
                scores = np.matmul(doc_vectors, query_vector)
                for doc, score in zip(copy.deepcopy(documents), scores):
                    doc['query_id'] = query_id
                    doc['score'] = round(float(score), 3)
                    result.append(doc)
                result.sort(key=lambda x: x['score'], reverse=True)
                for rank, doc in enumerate(result, start=1):
                    doc['rank'] = rank
                batch_results.append(result)
            all_results.extend(batch_results)
        return all_results


class Reranker:
    def __init__(self,
                 model_name_or_path: str,
                 batch_size: int,
                 cutoff_layers: int = 28,
                 device: int = 0):
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.cutoff_layers = cutoff_layers
        self.device = device
        self.reranker = self.init_reranker()

    def init_reranker(self):
        if 'layerwise' in self.model_name_or_path:
            return LayerWiseFlagLLMReranker(self.model_name_or_path, use_fp16=True, device=self.device)
        else:
            return FlagReranker(self.model_name_or_path, use_fp16=True, device=self.device)

    def rerank(self,
               queries: List[str],
               documents: List[List[Dict[str, Any]]],
               query_ids: List[str] = None
               ) -> List[List[Dict[str, Any]]]:
        assert len(documents) == len(queries)
        if query_ids is None:
            query_ids = [str(idx) for idx in range(len(queries))]
        assert len(queries) == len(query_ids)
        query_doc_pairs = []
        for query, docs in zip(queries, documents):
            for doc in docs:
                query_doc_pairs.append((query, doc['text']))
        score_list = self.reranker.compute_score(query_doc_pairs, batch_size=self.batch_size)
        if isinstance(score_list, float):
            score_list = [score_list]
        # for index in range(0, len(query_doc_pairs), self.batch_size):
        #     batch_query_doc_pairs = query_doc_pairs[index: index + self.batch_size]
        #     if 'layerwise' in self.model_name_or_path:
        #         batch_scores = self.reranker.compute_score(batch_query_doc_pairs, cutoff_layers=[self.cutoff_layers])
        #     else:
        #         batch_scores = self.reranker.compute_score(batch_query_doc_pairs, batch_size=self.batch_size)
        #     score_list.extend(batch_scores)
        results = copy.deepcopy(documents)
        count = 0
        for query_id, result in zip(query_ids, results):
            for doc, score in zip(result, score_list[count: count + len(result)]):
                doc['score'] = score
            result.sort(key=lambda x: x['score'], reverse=True)
            for rank, doc in enumerate(result, start=1):
                doc['rank'] = rank
            count += len(result)
        return results
