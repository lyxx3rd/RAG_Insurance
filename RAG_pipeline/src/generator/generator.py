import time
from copy import deepcopy
from typing import Dict, List, Union

from torch import bfloat16
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI, AzureOpenAI


class Generator:
    def __init__(self, model_name):
        self.model_name = model_name


class MistralGenerator(Generator):
    def __init__(self,
                 model_name,
                 device,
                 max_new_tokens=512,
                 temperature=0.7,
                 top_p=1.0,
                 frequency_penalty=0.0,
                 presence_penalty=0.0):
        super().__init__(model_name)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          load_in_8bit=False,
                                                          device_map="auto",
                                                          torch_dtype=bfloat16).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       padding_side='left',
                                                       use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self,
                 system_prompt: Union[str, List[str]],
                 user_prompt: Union[str, List[str]],
                 history: List[Dict[str, str]] = None):
        if history is None:
            history = []
        else:
            history = deepcopy(history)
        # messages = [{"role": "system", "content": system_prompt}]
        messages = []
        messages.extend(history)
        all_encoded = []
        for sp, up in zip(system_prompt, user_prompt):
            curr_message = messages + [{"role": "user", "content": up}]
            curr_message[0]['content'] = sp + curr_message[0]['content']
            encoded = self.tokenizer.apply_chat_template(
                conversation=curr_message,
                add_generation_prompt=True,
                tokenize=False)
            all_encoded.append(encoded)
        all_encoded = self.tokenizer(all_encoded,
                                     return_tensors="pt",
                                     padding=True)
        model_inputs = all_encoded.to(self.device)
        generated_ids = self.model.generate(**model_inputs,
                                            pad_token_id=self.tokenizer.eos_token_id,
                                            max_new_tokens=self.max_new_tokens,
                                            temperature=self.temperature,
                                            top_p=self.top_p,
                                            # frequency_penalty=self.frequency_penalty,
                                            # presence_penalty=self.presence_penalty,
                                            do_sample=True)
        all_decoded = self.tokenizer.batch_decode(generated_ids[:, model_inputs['input_ids'].shape[1]:],
                                                  skip_special_tokens=True)
        return all_decoded


class OpenAIGenerator(Generator):
    def __init__(self,
                 model_name: str,
                 is_chat_model: bool = True,
                 base_url: str = None,
                 api_key: str = None,
                 n: int = 1,
                 max_new_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 1.0,
                 frequency_penalty: float = 0.0,
                 presence_penalty: float = 0.0,
                 stop: str = '\n\n\n',
                 retry_times: Union[int, bool] = 0):
        super().__init__(model_name)
        self.is_chat_model = is_chat_model
        self.n = n
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.retry_times = retry_times
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def parse_response(self, response):
        to_return = []
        for _, choice in enumerate(response.choices):
            if self.is_chat_model:
                text = choice.message.content
                to_return.append((text, 0))
            else:
                text = choice.text
                logprob = sum(choice['logprobs']['token_logprobs'])
                to_return.append((text, logprob))
        texts = [r[0] for r in sorted(to_return, key=lambda tup: tup[1], reverse=True)]
        return texts

    def generate(self, system_prompt: str, user_prompt: str, history: List[Dict[str, str]] = None):
        if history is None:
            history = []
        if isinstance(user_prompt, list):
            user_prompt = user_prompt[0]
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})
        retry_times = self.retry_times
        get_results = False
        while not get_results:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    n=self.n,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    stop=self.stop
                )
                get_results = True
            except Exception:
                print(Exception)
                if retry_times > 0:
                    retry_times -= 1
                    time.sleep(1)
                else:
                    return [""] * self.n
        return self.parse_response(response)


class AzureOpenaiGenerator(OpenAIGenerator):
    def __init__(self,
                 model_name: str,
                 api_key: str,
                 endpoint: str,
                 api_version: str,
                 n: int = 1,
                 max_new_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 1.0,
                 frequency_penalty: float = 0.0,
                 presence_penalty: float = 0.0,
                 stop: str = None,
                 retry_times: Union[int, bool] = False):
        super().__init__(model_name, api_key, n, max_new_tokens, temperature, top_p, frequency_penalty, presence_penalty, stop, retry_times)
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )


# def generate_answer(data_path, output_path, start_index=0, end_index=-1):
#     generator = OpenAIGenerator(model_name='Llama-2-7b-chat-hf',
#                                 base_url='http://172.26.1.155:9005/v1',
#                                 api_key='none',
#                                 temperature=0.1,
#                                 top_p=0.6,
#                                 max_new_tokens=512)
#     with open(data_path, 'r') as f:
#         samples = json.load(f)
#     if end_index == -1:
#         end_index = len(samples) - 1
#     assert 0 <= start_index <= end_index <= len(samples)
#     samples = samples[start_index: end_index + 1]
#     with open(output_path, 'w') as f:
#         for sample in tqdm(samples, desc='Generating Query'):
#             response = generator.generate(system_prompt=sample['system'], user_prompt=sample['instruction'])[0].strip()
#             line = {'label': sample['output'], 'predict': response}
#             f.write(json.dumps(line, ensure_ascii=False) + '\n')
#             f.flush()
