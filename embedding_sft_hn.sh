python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
--model_name_or_path ./Model/bge-m3 \
--input_file ./data/embedding_sft.jsonl \
--output_file ./data/embedding_sft_minedHN.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--use_gpu_for_searching 