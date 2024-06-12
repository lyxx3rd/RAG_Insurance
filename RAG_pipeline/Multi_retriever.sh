python -m pyserini.search.lucene \
  --index ./indexes/Insurance/ \
  --topics ./data/qa_data.tsv \
  --output ./output/test.txt \
  --language zh \
  --bm25