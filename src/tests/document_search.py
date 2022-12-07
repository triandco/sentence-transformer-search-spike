from ..libs import get_content, get_files, search, quickTick, unzip
from ..custom_types import TestCase, RankResult
from ..embeddings import AbstractEmbeddingStrategy, WholeDocument, NthBlockDot, NthBlockCosine
from ..sentence_transformer import SentenceTransformerSpecb
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25L, BM25Plus
import sys
from time import perf_counter

making_sense_test_cases: 'list[TestCase]' = [
  {
    'query':'Sam talks about a scale of suffering to flourishing. His guest has seems to evaluate things from suffering to zero, Sam Harris seems to evaluate things from suffering to zero to flourishing.',
    'expect': 'Making_Sense_107_Is_Life_Actually_Worth_Living_Full_7-6-22'
  },
  {
    'query':'His guest has seems to evaluate things from suffering to zero, Sam Harris seems to evaluate things from suffering to zero to flourishing.',
    'expect': 'Making_Sense_107_Is_Life_Actually_Worth_Living_Full_7-6-22'
  },
  {
    'query':'Oh yeah, this reminds me of some Sammy podcast where he speaks to someone who has the view that life is a scale like: -1 —— 0 as in theres suffering or there’s not. Where as Sammy was viewing more like: -1 —— 0 —— +1 where there’s flourishing to be had.',
    'expect': 'Making_Sense_107_Is_Life_Actually_Worth_Living_Full_7-6-22'
  },
  {
    'query':'suffering scale',
    'expect': 'Making_Sense_107_Is_Life_Actually_Worth_Living_Full_7-6-22'
  },
  {
    'query':'Sam Harris suffering scale',
    'expect': 'Making_Sense_107_Is_Life_Actually_Worth_Living_Full_7-6-22'
  },
  {
    'query':'suffering flourishing scale',
    'expect': 'Making_Sense_107_Is_Life_Actually_Worth_Living_Full_7-6-22'
  },
  {
    'query':'guest evaluates things from suffering to zero, Sam evaluates things from suffering to zero to flourishing',
    'expect': 'Making_Sense_107_Is_Life_Actually_Worth_Living_Full_7-6-22'
  },
  {
    'query':'suffering to zero to flourishing',
    'expect': 'Making_Sense_107_Is_Life_Actually_Worth_Living_Full_7-6-22'
  }
]

not_boring_podcast_test_cases: 'list[TestCase]' = [
  {
    'query': "3 approaches of Capital",
    'expect': "capital-and-taste"
  },
  {
    'query': "a controversy when they sold Muslim prayer mats as fun home decor",
    'expect': "shein-the-tik-tok-ecomerce"
  },
  {
    'query': "Extending excellent taste",
    'expect': "capital-and-taste"
  },
  {
    'query': "combining usability with flexibility is both incredibly difficult and incredibly rewarding",
    'expect': "excel-never-dies"
  },

  {
    'query': "Investing in Ethereum",
    'expect': "own-the-internet"
  },
  {
    'query': 'best practices for DAOs',
    'expect': 'the-web3-debate.txt'
  },
  {
    'query': 'companies which Not Boring Captial has invested in',
    'expect': 'web3-usecase-of-the-future'
  },
  {
    'query': 'Which blockchain ecosystem is thriving and has a lot of offer dApp developer',
    'expect': 'solana-summer'
  }
]

def embeddings_rank(test_cases: 'list[TestCase]', documents: 'list[tuple[str, str]]', strategy: AbstractEmbeddingStrategy, verbose_log=True) -> 'list[RankResult]':  
  titles, docs = unzip(documents)
  embedding_generation_start = perf_counter()
  document_embeddings = [ strategy.document(document) for document in docs]
  embedding_generation_end = perf_counter()
  if verbose_log:
    print("⌚ Generation time {:f}s".format(embedding_generation_end - embedding_generation_start))
  outcome: 'list[RankResult]' = []

  for c in test_cases:
    query_embedding = strategy.query(c['query'])
    scores = [strategy.score(query_embedding, doc_embeddings) for doc_embeddings in document_embeddings]
    outcome.append({
      'case': c,
      'result': sorted(list(zip(titles, scores)), key=lambda x: x[1], reverse=True)
    })

  return outcome
  

def bm25_rank(test_cases: 'list[TestCase]', documents: 'list[tuple[str,str]]', strategy):
  titles, docs = unzip(documents)
  bm25l = strategy([document.split(' ') for document in docs])
  size = sys.getsizeof(bm25l.corpus_size) + sys.getsizeof(bm25l.doc_freqs) + sys.getsizeof(bm25l.idf)
  print("total size of indexed corpus %f " %  size /1000)
  outcome: 'list[RankResult]' = []
  for c in test_cases:
    query = c['query'].split()
    scores = bm25l.get_scores(query)
    
    
    outcome.append({
      'case': c,
      'result': sorted(list(zip(titles, scores)), key=lambda x:x[1], reverse=True)
    }) 
  
  return outcome


def tensor_to_score(tensor: torch.Tensor) -> float: 
  return tensor.cuda('cuda:0')[0].tolist()[0]


def test_embedding_rank(cases, documents, directory_path, verbose=True):
  base_model_name = 'multi-qa-mpnet-base-dot-v1'
  model_base = SentenceTransformer('sentence-transformers/%s' % base_model_name)
  model_dot_128 = SentenceTransformer('models/reduced-128/%s' % base_model_name)
  model_dot_256 = SentenceTransformer('models/reduced-256/%s' % base_model_name)

  normalise_embeddings = True
  embeddings_strategies = [ 
    NthBlockDot(model_base, '%s-768' % base_model_name, 1, normalise_embeddings),
    NthBlockDot(model_dot_128, '%s-128' % base_model_name, 3, normalise_embeddings),
    NthBlockDot(model_dot_256, '%s-256' % base_model_name, 2, normalise_embeddings),
  ]

  for strategy in embeddings_strategies:
    print('Strategy ', strategy.__class__.__name__)
    success_count = 0
    for result in embeddings_rank(cases, documents, strategy):
      success = result['case']['expect'] in result['result'][0][0]
      if success:
        success_count += 1
      print(' ', quickTick(success), 'Query: ', result['case']['query'])
      print('  Expected: ', result['case']['expect'])
      if verbose:
        for doc, score in result['result']:
          print('   ', score, doc.replace(directory_path, '').replace('.txt',''))
        print(' ')
    print('⚙ ', strategy)
    print('👉  Success Rate {:s} {:d}/{:d}'.format(strategy.__class__.__name__, success_count, len(cases)))
    print(' ')
    print(' ')


def test_bm25_rank(cases, documents, directory_path):
  bm_strategies = [ BM25L, BM25Plus ]

  for strategy in bm_strategies:
    print('Strategy ', strategy.__name__)
    success_count = 0
    for result in bm25_rank(cases, documents, strategy):
      success = result['case']['expect'] in result['result'][0][0]
      if success:
        success_count += 1
      print(' ', quickTick(success), 'Query: ', result['case']['query'])
      print('  Expect: ', result['case']['expect'])
      for doc, score in result['result']:
        print('   ', score, doc.replace(directory_path, '').replace('.txt',''))
      print(' ')
    print('👉 Success Rate {:s} {:d}/{:d}'.format(strategy.__name__, success_count, len(cases)))


def run_test(cases, directory_path):
  files = get_files(directory_path)
  documents = list(zip(files, list(map(get_content, files))))

  test_bm25_rank(cases, documents, directory_path)
  # test_embedding_rank(cases, documents, directory_path, True)  


if __name__=='__main__':
  run_test(not_boring_podcast_test_cases,'doc/not-boring-podcast')
  # run_test(making_sense_test_cases,'doc/making-sense')
  
  