from libs import get_files, search
from embeddings import AbstractEmbeddingEncoder, LongSentence512, Mean, AMaxParagraph

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

def run_test(strategy: AbstractEmbeddingEncoder):  
  directory_path = 'doc/not-boring-podcast'
  files = get_files(directory_path)
  corpus_embedding = [ strategy.document(file_path) for file_path in files]

  success_count = 0
  for c in test_cases:
    query_embedding = strategy.query(c['query'])
    result = search(query_embedding, corpus_embedding)

    # if verbose_log:
    #   for hit in result:
    #     print('*', files[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
    first_match = files[result[0]['corpus_id']]
    success = c['expect'] in first_match
    if success:
      success_count += 1

    print('✅' if success else '❌', "Case: {:s}".format(c['title']))
    print("Query: '{:s}'".format(c['query']))
    print('Expected first match: {:s}. Found {:s}'.format(c['expect'], first_match.replace('doc/not-boring-podcast', '')) )
    print("-------------------------------------------------")

  return success_count / len(test_cases) * 100
  


if __name__=='__main__':
  strategies = [
    # {
    #   'name': 'Long sentence',
    #   'implementation':LongSentence512
    # },
    {
      'name': 'Mean',
      'implementation': Mean
    },
    # {
    #   'name': 'Numpy Amax - paragraph',
    #   'implementation': AMaxParagraph
    # },
    # {
    #   'name': 'Numpy Sum Concatted with Amax - Sentence',
    #   'implementation': SumAMaxSentence
    # }
  ]

  for strategy in strategies:
    success_rate = run_test(strategy['implementation'])
    print(strategy['name'],"{:10.2f}%".format(success_rate), '👍' if success_rate >= 70 else '👎',)