from ..libs import get_content
from sentence_transformers import SentenceTransformer
from ..embeddings import AbstractEmbeddingEncoder, Mean, AMax, MeanAMax

def sammy_podcast_excerpts():
  docs = get_content('doc/sammy.txt')
  outcome = filter(lambda x: (x != '\n'), docs)
  return list(outcome)

def not_boring_podcast():
  docs = [
    'capital-and-taste',
    'solana-summer',
    'the-web3-debate',
    'optimism',
    'the-great-online-game',
  ]
  paths = list(map(lambda x: 'doc/not-boring-podcast/{:s}.txt'.format(x), docs))
  content = list(map(get_content, paths))
  return content

def get_test_cases():
  return [
    # {
    #   'documents': sammy_podcast_excerpts(),
    #   'query': 'guest evaluates things from suffering to zero, Sam evaluates things from suffering to zero to flourishing'
    # },
    {
      'documents': not_boring_podcast(),
      'query': '3 approaches of Capitals'
    }
  ]

def run_test(case):
  model = SentenceTransformer('msmarco-distilbert-base-tas-b')

  strategies: 'list[AbstractEmbeddingEncoder]' = [
    Mean(model),
    AMax(model), 
    MeanAMax(model)
  ]
  for strategy in strategies:
    document_embeddings = list(map(strategy.document, case['documents']))
    query_embedding = strategy.query(case['query'])
    scores = [ strategy.score(query_embedding, embedding).cuda('cuda:0')[0].tolist() for embedding in document_embeddings ]
    print('Strategy', strategy.__name__)
    for (index, score) in enumerate(scores):
      print('doc[{:d}]'.format(index), score)
    print('----------')

  
if __name__ == '__main__':
  cases = get_test_cases()
  for case in cases: 
    run_test(case)
  
 
    

  