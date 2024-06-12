torchrun --nproc_per_node 2 \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir ./Model/Sft_model \
--model_name_or_path ./Model/bge-m3 \
--train_data ./data/embedding_sft_minedHN.jsonl \
--learning_rate 1e-5 \
--fp16 True \
--num_train_epochs 5 \
--per_device_train_batch_size 1 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 32 \
--passage_max_len 128 \
--train_group_size 2 \
--negatives_cross_device \
--logging_steps 10 \
--save_steps 1000 \
--query_instruction_for_retrieval "" 