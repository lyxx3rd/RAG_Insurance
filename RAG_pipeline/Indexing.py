import subprocess
from .self_function import prepare_directory, check_dict_and_modify_time

def check_indexing():
    i = check_dict_and_modify_time(RAG_data_modification_time_path = "./conf/RAG_data_modification_time.json",
               path_list = ["./RAG_data/QA_json","./RAG_data/QA_json","./RAG_data/Contract_json","./RAG_data/Contract_jsonl"])
    if i == 0:
        print("数据库已是最新，不更新Indexing")
        return
    else:
        output_path = "./indexes/Insurance/bm25_index/QA_index"
        prepare_directory(output_path)
        command_bm25_qa = f"""
        python -m pyserini.index.lucene \
          --collection JsonCollection \
          --input ./RAG_data/QA_json \
          --index {output_path} \
          --generator DefaultLuceneDocumentGenerator \
          --threads 1 \
          --storePositions --storeDocvectors --storeRaw
        """
        
        output_path = "./indexes/Insurance/bm25_index/Contract_index"
        prepare_directory(output_path)
        command_bm25_contract = f"""
        python -m pyserini.index.lucene \
          --collection JsonCollection \
          --input ./RAG_data/Contract_json \
          --index {output_path} \
          --generator DefaultLuceneDocumentGenerator \
          --threads 1 \
          --storePositions --storeDocvectors --storeRaw
        """

        output_path = "./indexes/Insurance/dense_index/QA_index"
        prepare_directory(output_path)
        command_dense_qa = f"""
        python -m pyserini.encode \
          input   --corpus ./RAG_data/QA_jsonl/QA_list.jsonl \
                  --fields text \
                  --delimiter None \
                  --shard-id 0 \
                  --shard-num 1 \
          output  --embeddings {output_path} \
                  --to-faiss \
          encoder --encoder ./Model/mixed_model_1 \
                  --fields text \
                  --batch 16 \
                  --max-length 512 \
                  --dimension 1024 \
                  --pooling 'cls' \
                  --device 'cuda:0' \
                  --l2-norm \
                  --fp16
        """
        
        output_path = "./indexes/Insurance/dense_index/Contract_index"
        prepare_directory(output_path)
        command_dense_contract = f"""
        python -m pyserini.encode \
          input   --corpus ./RAG_data/Contract_jsonl/Contract_list.jsonl \
                  --fields text \
                  --delimiter None \
                  --shard-id 0 \
                  --shard-num 1 \
          output  --embeddings {output_path} \
                  --to-faiss \
          encoder --encoder ./Model/mixed_model_1 \
                  --fields text \
                  --batch 16 \
                  --max-length 512 \
                  --dimension 1024 \
                  --pooling 'cls' \
                  --device 'cuda:0' \
                  --l2-norm \
                  --fp16
        """
        
        
        command_list = [command_bm25_qa,command_bm25_contract,command_dense_qa,command_dense_contract]
        for command in command_list:
            try:
                subprocess.run(command, shell=True, check=True,stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                print(f"Indexing 生成成功.")
            except subprocess.CalledProcessError as e:
                print(f"Error occurred: {e}")
    print("数据库更新完毕")
