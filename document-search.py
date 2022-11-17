from sentence_transformers import SentenceTransformer, util
from libs import all_lines, get_files
model = SentenceTransformer('msmarco-distilbert-base-tas-b')

test_cases = [
  {
    'title': "Exact wording",
    'query': "Livestream and entertainment",
    'expect': "shein-the-tik-tok-ecomerce"
  },
  {
    'title': "Exact wording",
    'query': "Extending excellent taste",
    'expect': "capital-and-taste"
  },
  {
    'title': "Exact quote",
    'query': "combining usability with flexibility is both incredibly difficult and incredibly rewarding",
    'expect': "excel-never-dies"
  },
  {
    'title': "Single point",
    'query': 'best practices for DAOs',
    'expect': 'the-web3-debate.txt'
  },
  {
    'title': "Single point",
    'query': "3 approaches of Capital",
    'expect': "capital-and-taste"
  },
  {
    'title': "Single point",
    'query': "Ethereum explanation",
    'expect': "own-the-internet"
  },
  {
    'title': "Single point",
    'query': 'companies which Not Boring Captial has invested in',
    'expect': 'web3-usecase-of-the-future'
  }
]
  
def search(query, corpus):
  query_embedding = model.encode(query, convert_to_tensor=True)
  hits = util.semantic_search(query_embedding, corpus, top_k=3, score_function=util.dot_score)
  return hits[0]

if __name__=='__main__':
  directory_path = 'doc/not-boring-podcast'
  files = get_files(directory_path)
  corpus = [ "\n".join(all_lines(doc)) for doc in files]
  corpus_embedding = model.encode(corpus, convert_to_tensor=True)
  
  success_count = 0
  for c in test_cases:
    result = search(c['query'], corpus_embedding)

    # if verbose_log:
    #   for hit in result:
    #     print('*', files[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
    first_match = files[result[0]['corpus_id']]
    success = c['expect'] in first_match
    if not success:
      success_count += 1

    print('✅' if success else '❌', "Case: {:s}".format(c['title']))
    print("Query: '{:s}'".format(c['query']))
    print('Expected first match: {:s}. Found {:s}'.format(c['expect'], first_match.replace('doc/not-boring-podcast', '')) )
    print("-------------------------------------------------")

  success_rate = success_count / len(test_cases) * 100
  print("Success rate: {:10.2f}%".format(success_rate), '👍' if success_rate >= 70 else '👎',)