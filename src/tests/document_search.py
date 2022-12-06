from ..libs import get_content, get_files, search, quickTick, unzip
from ..custom_types import TestCase, RankResult
from ..embeddings import AbstractEmbeddingStrategy, Mean, AMax, MeanAMax, DocumentParagraph, NthBlockDot, NthBlockCosine
from ..sentence_transformer import SentenceTransformerSpecb
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25L, BM25Plus
from time import perf_counter

test_cases: 'list[TestCase]' = [
  {
    'title': "Single point",
    'query': "3 approaches of Capital",
    'expect': "capital-and-taste"
  },
  {
    'title': "Exact wording",
    'query': "a controversy when they sold Muslim prayer mats as fun home decor",
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

def embeddings_rank(documents: 'list[tuple[str, str]]', strategy: AbstractEmbeddingStrategy, verbose_log=True) -> 'list[RankResult]':  
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
  

def bm25_rank(documents: 'list[tuple[str,str]]', strategy):
  titles, docs = unzip(documents)
  bm25l = strategy([document.split(' ') for document in docs])
  
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


def test_embedding_rank(documents, verbose=True):
  
  modelDot256 = SentenceTransformer('models/reduced-256/msmarco-bert-base-dot-v5')
  # model128 = SentenceTransformer('models/reduced-128/msmarco-distilbert-cos-v5')
  # model256 = SentenceTransformer('models/reduced-256/msmarco-distilbert-cos-v5')
  # model512 = SentenceTransformer('models/reduced-512/msmarco-distilbert-cos-v5')
  # model2 = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5')
  # model3 = SentenceTransformer('sentence-transformers/msmarco-distilbert-cos-v5')
  # modelSpecb = SentenceTransformerSpecb("Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit")

  embeddings_strategies = [ 
    # MeanAMax(model),
    NthBlockDot(modelDot256, 3),
    # AMax(model),
    # Mean(model),
    # SGPTCosine(modelSpecb),  
    # DocumentParagraph(model)
  ]

  for strategy in embeddings_strategies:
    print('Strategy ', strategy.__class__.__name__)
    success_count = 0
    for result in embeddings_rank(documents, strategy):
      success = result['case']['expect'] in result['result'][0][0]
      if success:
        success_count += 1
      print(' ', quickTick(success), 'Query: ', result['case']['query'])
      print('  Expected: ', result['case']['expect'])
      if verbose:
        for doc, score in result['result']:
          print('   ', score, doc.replace(directory_path, '').replace('.txt',''))
        print(' ')
    print('👉  Success Rate {:s} {:d}/{:d}'.format(strategy.__class__.__name__, success_count, len(test_cases)))
    print(' ')
    print(' ')


def test_bm25_rank(documents):
  bm_strategies = [ BM25L, BM25Plus ]

  for strategy in bm_strategies:
    print('Strategy ', strategy.__name__)
    success_count = 0
    for result in bm25_rank(documents, strategy):
      success = result['case']['expect'] in result['result'][0][0]
      if success:
        success_count += 1
      print(' ', quickTick(success), 'Query: ', result['case']['query'])
      print('  Expect: ', result['case']['expect'])
      for doc, score in result['result']:
        print('   ', score, doc.replace(directory_path, '').replace('.txt',''))
      print(' ')
    print('👉 Success Rate {:s} {:d}/{:d}'.format(strategy.__name__, success_count, len(test_cases)))


if __name__=='__main__':
  directory_path = 'doc/not-boring-podcast'
  files = get_files(directory_path)
  documents = list(zip(files, list(map(get_content, files))))

  # test_bm25_rank(documents)
  test_embedding_rank(documents, False)  
  
  