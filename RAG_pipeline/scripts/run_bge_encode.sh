python -m pyserini.encode \
  input   --corpus  ../RAG_data/Contract_list/Contract_list.jsonl \
          --fields text \
          --delimiter None \
          --shard-id 0 \
          --shard-num 1 \
  output  --embeddings ../indexes/bge/Contract \
          --to-faiss \
  encoder --encoder ../../Model/bge-m3 \
          --fields text \
          --batch 16 \
          --max-length 512 \
          --dimension 1024 \
          --pooling 'cls' \
          --device 'cuda:0' \
          --l2-norm \
          --fp16
