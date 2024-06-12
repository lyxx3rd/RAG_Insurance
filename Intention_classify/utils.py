import dashscope
dashscope.api_key="sk-632f5cf28f0a43719096801cd7c2e61a"
from http import HTTPStatus
import dashscope
from torch.utils.data import Dataset
from dataclasses import dataclass

def Call_with_messages(messages):
    response = dashscope.Generation.call(
        dashscope.Generation.Models.qwen_turbo,
        messages=messages,
        result_format='message',  # 将返回结果格式设置为 message
    )
    content = response.output.choices[0]["message"]["content"]
    return content

@dataclass
class DataItem:
    sentence: str
    label: int

class DatasetClassify(Dataset):
    def __init__(self, question):
        self.data_list = [{"sentence":question,"label":0}]

    def __getitem__(self, index):
        item = self.data_list[index]
        item = DataItem(**item)
        content = item.sentence
        return content, item.label

    def __len__(self):
        return len(self.data_list)
