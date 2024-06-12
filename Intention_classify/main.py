from user import user_chat

tokenizer_model_path = "./Model/chinese-roberta-wwm-ext"
classify_model_path="./Model_save/classify_model.pt"
user_chat(classify_model_path,tokenizer_model_path)