from QA_math import QA_math
from Contract_search import Qwen_search
from lunch_process import lunch_process

print("开始加载程序")
hsearcher,contract_dict,QA_dict,reranker = lunch_process()

question = ""
while True:
    if question == "":
        question = input("请输入测试问题:")
    ide, q, a = QA_math_Qwen(QA_dict, question)
    if ide == 0:
        print("根据您输入的问题,您可以参考一下问题以及回答")
        print("[参考问题]:",q)
        print("[参考回答]:",a)
        question = ""
    if ide == 1:
        print("启动条款匹配程序")
        res, item1, item2, item3 = Qwen_search(question,hsearcher,contract_dict)
        print(res)
        question = input("如需显示具体条款请输入[显示条款],或直接输入新问题:")
        if question == "显示条款":
            print(item1,"\n")
            print(item2,"\n")
            print(item3,"\n")
            question == ""
            question = input("您还可以输入其他的咨询问题:")