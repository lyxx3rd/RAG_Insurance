import dashscope
from http import HTTPStatus
dashscope.api_key="sk-632f5cf28f0a43719096801cd7c2e61a"

def call_with_decomposition(question,item1,item2,item3):
    messages = "客户的问题是：\n" + question + "\n根据问题所匹配的条例为1：\n" + item1 + "\n根据问题所匹配的条例为2：\n" + item2 + "\n根据问题所匹配的条例为3：\n" + item3
    messages = [{'role': 'system', 'content': '你是一个医疗保险条例解读专家'},
                {'role': 'system', 'content': '请为用户根据下面给出的保险条例来回答客户的问题'},
                {'role': 'user', 'content': messages}]

    response = dashscope.Generation.call(
        dashscope.Generation.Models.qwen_turbo,
        messages=messages,
        result_format='message',  # 将返回结果格式设置为 message
    )
    content = response.output.choices[0]["message"]["content"]
    return content

def call_with_simple(question):
    content = "我在进行一个文字检索任务，请帮我从客户给出的问题中提取三个医疗保险咨询相关的关键词用于条文检索。\n 请注意，需要忽略如经典个人计划A产品和医疗保险等产品名。\n 请注意，只回答我关键词，不要其他内容。\n 【具体问题如下】：\n" + question
    messages = [{'role': 'system', 'content': '你是一个有用的助手'},
                {'role': 'user', 'content': content}]

    response = dashscope.Generation.call(
        model='qwen-max-0428',
        messages=messages
    )
    return response["output"]["text"]

def Qwen_search(question,hsearcher,contract_dict):
    question_word = call_with_simple(question)
    hits = hsearcher.search(question_word, k=3, alpha=0.3)
    item1 = contract_dict[int(hits[0].docid)]['contents']
    item2 = contract_dict[int(hits[1].docid)]['contents']
    item3 = contract_dict[int(hits[2].docid)]['contents']
    ans = call_with_decomposition(question, item1, item2, item3)
    return ans, item1, item2, item3