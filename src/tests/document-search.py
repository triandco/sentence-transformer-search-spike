from ..libs import get_content, get_files, search, quickTick
from ..embeddings import AbstractEmbeddingEncoder, Default, MeanAMax

test_cases = [
  {
    'title': "Single point",
    'query': "3 approaches of Capital",
    'expect': "capital-and-taste"
  },
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
    'title': "Paraphrase content",
    'query': "Investing in Ethereum",
    'expect': "own-the-internet"
  },
  {
    'title': "Single point",
    'query': 'best practices for DAOs',
    'expect': 'the-web3-debate.txt'
  },
  {
    'title': "Details inside documents",
    'query': 'companies which Not Boring Captial has invested in',
    'expect': 'web3-usecase-of-the-future'
  },
  {
    'title': 'Details at the end of document',
    'query': 'Which blockchain ecosystem is thriving and has a lot of offer dApp developer',
    'expect': 'solana-summer'
  }
]

def run_test(documents: 'list[str]', strategy: AbstractEmbeddingEncoder, verbose_log=True):  
  
  document_embeddings = [ strategy.document(document) for document in documents]

  success_count = 0
  for c in test_cases:
    query_embedding = strategy.query(c['query'])
    result = search(query_embedding, document_embeddings)
    
    matches = list(map(lambda x: files[x['corpus_id']], result))
    success = c['expect'] in matches[0]
    if success:
      success_count += 1
    
    if verbose_log:
      print(quickTick(success), "Case: {:s}".format(c['title']))
      print("Query: '{:s}'".format(c['query']))
      print('Match found', matches)
      print('Expected first match: {:s}.'.format(c['expect']) )
      print("-------------------------------------------------")

  return success_count / len(test_cases) * 100
  


if __name__=='__main__':
  directory_path = 'doc/not-boring-podcast'
  files = get_files(directory_path)
  documents = list(map(get_content, files))

  strategies = [ Default, MeanAMax ]

  for strategy in strategies:
    run_test(documents, strategy, verbose_log=True)