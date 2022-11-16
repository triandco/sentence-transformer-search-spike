from sentence_transformers import SentenceTransformer, util
from libs import read_file
model = SentenceTransformer('msmarco-distilbert-base-tas-b')


lines = read_file('doc/sammy.txt')
query_embedding = model.encode('suffering to zero to flourishing', convert_to_tensor=True)
passages_embedding = model.encode(lines, convert_to_tensor=True)

print("build completed.")
hits = util.semantic_search(query_embedding, passages_embedding, top_k=3, score_function=util.dot_score)
result = hits[0]
print("search complete.")
print("Found {:d}".format(len(result)))
for hit in result:
  print('')
  print(hit['corpus_id'], lines[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
  print('')
