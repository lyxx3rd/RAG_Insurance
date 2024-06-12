
import re
import time
import traceback

import math
import numpy as np
from flask import Flask, request, jsonify

from .utils import retriever, reranker, logger, app_conf, k, query_to_pos
from .. import SparseRetriever


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
app.json.ensure_ascii = False
logger.info('Server Start')
logger.info(f"'model_name_or_path: {app_conf['retrieve_model']['model_name_or_path']}'")
logger.info(f"'index_dir: {app_conf['corpus']['index_dir']}'")
logger.info(f"'pooling: {app_conf['retrieve_model']['pooling']}'")
logger.info(f"'l2_norm: {app_conf['retrieve_model']['l2_norm']}'")
logger.info(f"'query_prefix: {app_conf['query']['prefix']}'")
logger.info(f"'ef_search: {app_conf['retrieve_model']['ef_search']}'")


@app.route("/vectorize_query", methods=["POST"])
def vectorize_query():
    if isinstance(retriever, SparseRetriever):
        raise TypeError(f'class "SparseRetriever" has no method "vectorize"')
    queries = request.json
    try:
        assert len(queries) > 0
        logger.info(f'Get {len(queries)} texts: {queries}')
        start = time.time()
        vectors = retriever.vectorize(texts=queries, is_query=True).tolist()
        infer_time = time.time() - start
        logger.info(f'average inference time {infer_time / len(queries):.3}s/text')
        logger.info(f'total inference time {infer_time:.3}s for {len(queries)} texts')
    except Exception as e:
        vectors = None
        logger.error(f'Error occurred at line {e.__traceback__.tb_lineno}:\n{traceback.format_exc()}')
    results = {
        'vectors': vectors,
    }
    return jsonify(results)


@app.route("/vectorize_document", methods=["POST"])
def vectorize_document():
    if isinstance(retriever, SparseRetriever):
        raise TypeError(f'class "SparseRetriever" has no method "vectorize"')
    documents = request.json
    try:
        assert len(documents) > 0
        logger.info(f'Get {len(documents)} texts: {documents}')
        start = time.time()
        vectors = retriever.vectorize(texts=documents, is_query=False).tolist()
        infer_time = time.time() - start
        logger.info(f'average inference time {infer_time / len(documents):.3}s/text')
        logger.info(f'total inference time {infer_time:.3}s for {len(documents)} texts')
    except Exception as e:
        vectors = None
        logger.error(f'Error occurred at line {e.__traceback__.tb_lineno}:\n{traceback.format_exc()}')
    results = {
        'vectors': vectors,
    }
    return jsonify(results)


@app.route("/retrieve", methods=["POST"])
def retrieve():
    queries = request.json
    try:
        assert len(queries) > 0
        start = time.time()
        if isinstance(queries[0], list):
            queries = np.array(queries)
            logger.info(f'Get {len(queries)} query vectors.')
        elif isinstance(queries[0], str):
            logger.info(f'Get {len(queries)} queries: {queries}')
        else:
            raise ValueError(f"Invalid input data: {queries[0]}")
        similarities = retriever.retrieve(queries=queries, k=k)
        infer_time = time.time() - start
        logger.info(f'average inference time {infer_time / len(queries):.3}s/query')
        logger.info(f'total inference time {infer_time:.3}s for {len(queries)} queries')
    except Exception as e:
        similarities = None
        logger.error(f'Error occurred at line {e.__traceback__.tb_lineno}:\n{traceback.format_exc()}')
    results = {
        'similarity': similarities,
    }
    return jsonify(results)


@app.route("/mix_retrieve", methods=["POST"])
def mix_retrieve():
    if isinstance(retriever, SparseRetriever):
        raise TypeError(f'class "SparseRetriever" has no method "mix_retrieve"')
    query_paris = request.json
    try:
        queries1, queries2 = query_paris[0], query_paris[1]
        mix_ratio = query_paris[2] if len(query_paris) == 3 else 0.5
        assert len(queries1) == len(queries2) > 0
        logger.info(f'Get {len(queries1)} query pairs.')
        start = time.time()
        if isinstance(queries1[0], list):
            queries1 = np.array(queries1)
        if isinstance(queries2[0], list):
            queries2 = np.array(queries2)
        similarities = retriever.mix_retrieve(queries1=queries1,
                                              queries2=queries2,
                                              k=k,
                                              mix_ratio=mix_ratio)
        infer_time = time.time() - start
        logger.info(f'average inference time {infer_time / len(queries1):.3}s/query pair')
        logger.info(f'total inference time {infer_time:.3}s for {len(queries1)} query pairs')
    except Exception as e:
        similarities = None
        logger.error(f'Error occurred at line {e.__traceback__.tb_lineno}:\n{traceback.format_exc()}')
    results = {
        'similarity': similarities,
    }
    return jsonify(results)


@app.route("/rerank", methods=["POST"])
def rerank():
    query_documents = request.json
    try:
        queries, documents = query_documents['queries'], query_documents['documents']
        assert len(queries) == len(documents) > 0
        logger.info(f'Get {len(queries)} query_document pairs for reranking.')
        start = time.time()
        if isinstance(queries[0], list):
            queries = np.array(queries)
        # similarities = retriever.rerank(queries=queries, documents=documents)
        similarities = reranker.rerank(queries=queries, documents=documents)
        infer_time = time.time() - start
        logger.info(f'average inference time {infer_time / len(queries):.3}s/query_document pair')
        logger.info(f'total inference time {infer_time:.3}s for {len(queries)} query_document pairs')
    except Exception as e:
        similarities = None
        logger.error(f'Error occurred at line {e.__traceback__.tb_lineno}:\n{traceback.format_exc()}')
    results = {
        'similarity': similarities,
    }
    return jsonify(results)


@app.route("/get_rewards", methods=["POST"])
def get_rewards():
    data = request.json
    try:
        messages = data['messages']
        reward_type = data['reward_type'] if 'reward_type' in data.keys() else 'relative_ranking'
        assert len(messages) > 0
        ori_queries = []
        rew_queries = []
        for idx in range(len(messages)):
            system_prompt_end = messages[idx].find('[INST]')
            # system_prompt_end = messages[idx].find('user')
            if system_prompt_end == -1:
                raise ValueError(f'Missing "[INST]" in input: {messages[idx]}')
            messages[idx] = messages[idx][system_prompt_end:]
        pattern = r'\[INST\]\s*(.+) \[/INST\]\s*(.+)'
        # pattern = r'user\n(.+)\nassistant\n(.+)'
        for message in messages:
            match = re.search(pattern, message, re.DOTALL)
            if match:
                ori_query = match.group(1).removeprefix('Original Query: ').removeprefix('Query: ')
                rew_query = match.group(2)
                start_index = rew_query.find('Rewritten Query: ')
                if start_index != -1:
                    rew_query = rew_query[start_index:].removeprefix('Rewritten Query: ')
            else:
                logger.warning(f'Invalid message: {message}')
                ori_query = rew_query = ""
            ori_queries.append(ori_query)
            rew_queries.append(rew_query)
        logger.info(f'Get {len(messages)} messages: ' +
                    ', '.join([f'{{message: "{message}", ori_query: "{ori_query}", rew_query: "{rew_query}"}}'
                               for message, ori_query, rew_query in zip(messages, ori_queries, rew_queries)]))
        start = time.time()
        if reward_type in {'absolute_ranking', 'relative_ranking'}:
            all_results = retriever.retrieve(queries=ori_queries+rew_queries,
                                             k=k)
        elif reward_type == 'mixed_relative_ranking':
            all_results = retriever.mix_retrieve(queries1=ori_queries+rew_queries,
                                                 queries2=ori_queries+ori_queries,
                                                 k=k)
        else:
            raise ValueError(f'Invalid reward type: {reward_type}')
        ori_results, rew_results = all_results[:len(ori_queries)], all_results[len(ori_queries):]
        if reward_type == 'absolute_ranking':
            clip_range = (0.0, 1.0)
            base_dcg = 1 / math.log(2 + int(1.5 * k), 3)
            def get_gcd(rank):
                return 1 / math.log(2 + rank, 3)
        else:
            clip_range = (-2, 2)
            base_dcg = 1 / math.log10(9 + int(1.5 * k))
            def get_gcd(rank):
                return 1 / math.log10(9 + rank)
        rewards = []
        for ori_query, ori_result, rew_result in zip(ori_queries, ori_results, rew_results):
            ori_dcg = rew_dcg = 0.0
            true_docids = query_to_pos.get(ori_query, None)
            if true_docids is not None:
                found_positives = 0
                for p1, p2 in zip(ori_result, rew_result):
                    if str(p1['docid']) in true_docids:
                        ori_dcg += get_gcd(int(p1['rank'])) / len(true_docids)
                        found_positives += 1
                    if str(p2['docid']) in true_docids:
                        rew_dcg += get_gcd(int(p2['rank'])) / len(true_docids)
                        found_positives += 1
                    if found_positives >= 2 * len(true_docids):
                        break
                if ori_dcg == 0.0:
                    ori_dcg = base_dcg
                if rew_dcg == 0.0:
                    rew_dcg = base_dcg
                if reward_type == 'absolute_ranking':
                    reward = round((rew_dcg - base_dcg) / (1 - base_dcg), 3)
                else:
                    reward = round((rew_dcg - ori_dcg) * 10, 3)
            elif ori_query == "":
                reward = clip_range[0]
            else:
                logger.warning(f'No label document found for this query: "{ori_query}"')
                reward = 0.0
            reward = min(max(reward, clip_range[0]), clip_range[1])
            rewards.append(reward)
        infer_time = time.time() - start
        logger.info(f'Rewards: {rewards}')
        logger.info(f'Average inference time {infer_time / len(rew_queries):.3}s/query')
        logger.info(f'Total inference time {infer_time:.3}s for {len(rew_queries)} queries')
    except Exception as e:
        rewards = None
        logger.error(f'Error occurred at line {e.__traceback__.tb_lineno}:\n{traceback.format_exc()}')
    results = {
        'scores': rewards,
    }
    return jsonify(results)


@app.route('/liveness', methods=['GET'])
def liveness():
    return jsonify("ok")


@app.route('/readiness', methods=['GET'])
def readiness():
    return jsonify("ok")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000)
