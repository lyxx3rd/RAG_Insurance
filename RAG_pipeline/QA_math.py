from .utils_bm25 import creat_retriever
import dashscope
from http import HTTPStatus

dashscope.api_key="sk-632f5cf28f0a43719096801cd7c2e61a"
def call_with_messages_qa(question,q,a):
    messages = "客户的问题是：\n" + question + "\n根据此问题我找到的参考问题是：\n" + q + "\n参考问题对应的答案是：\n" + a 
    messages = [{'role': 'system', 'content': '你是一个有用的助手'},
                {'role': 'system', 'content': '请协助我判断下列我找到的参考问题是否有效的回答了客户提出的问题, 请回答已解答或不相关三个单词'},
                {'role': 'user', 'content': messages}]

    response = dashscope.Generation.call(
        dashscope.Generation.Models.qwen_turbo,
        messages=messages,
        result_format='message',  # 将返回结果格式设置为 message
    )
    content = response.output.choices[0]["message"]["content"]
    return content

def similarities_QA(question,qa_dict,hsearcher_QA,k=1,usage_hsearcher=False):
    retriever = creat_retriever("QA")
    if usage_hsearcher == False:
        similarities = retriever.retrieve(queries=[question], k=k)
        mate_query = similarities[0]
    else:
        hits = hsearcher_QA.search(question, k=k, alpha=0.3)
        item_list=[]
        for i in range(k):
            try:
                item={}
                item['text'] = qa_dict[int(hits[i].docid)]['contents']
                item_list.append(item)
            except:
                print(i)
                print(int(hits[i].docid))
        mate_query = item_list
    return mate_query

def QA_math_Qwen(QA_answer_dict,qa_dict, question,hsearcher_QA):
    print("正在使用Qwen判断算法模型")
    similarities = similarities_QA(question,qa_dict,hsearcher_QA,k=1,usage_hsearcher=True)
    q = similarities[0]['text']
    a = QA_answer_dict[q]
    res = call_with_messages_qa(question,q,a)
    if res == "已解答":
        return 0, q, a
    elif res == "不相关":
        return 1, q, a

def QA_math_rerank(QA_answer_dict,qa_dict, question,reranker,hsearcher_QA=None):
    print("正在试用rerank算法模型")
    similarities = similarities_QA(question,qa_dict,hsearcher_QA,k=5,usage_hsearcher=True)
    q0 = similarities[0]['text']
    q1 = similarities[1]['text']
    q2 = similarities[2]['text']
    q3 = similarities[3]['text']
    q4 = similarities[4]['text']
    scores = reranker.compute_score([[question,q0],[question,q1],[question,q2],[question,q3],[question,q4]])
    max_value = max(scores)
    max_index = scores.index(max_value)
    print("最佳index是:",max_index+1)
    q = similarities[max_index]['text']
    a = QA_answer_dict[q]
    res = 0
    return 0, q, a



