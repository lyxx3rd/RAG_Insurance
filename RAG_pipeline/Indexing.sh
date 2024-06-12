python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ./RAG_data/Contract_list \
  --index ./indexes/Insurance/Contract_list \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw