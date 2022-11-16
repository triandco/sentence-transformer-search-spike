from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('msmarco-distilroberta-base-v2')

def read_file(path):
  content = ''
  # ignore emoji
  with open(path, encoding='utf-8', errors='ignore') as f:
    content = f.readlines()
  return content


lines = read_file('doc/taste.txt')
query_embedding = model.encode('extending its taste', convert_to_tensor=True)
passages_embedding = model.encode(lines, convert_to_tensor=True)


print("build completed.")
hits = util.semantic_search(query_embedding, passages_embedding, top_k=5, score_function=util.dot_score)
result = hits[0]
print("search complete.")
print("Found {:d}".format(len(result)))
for hit in result:
  print('```')
  print(lines[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
  print('```')
