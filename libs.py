from os import listdir
from os.path import isfile, join
from sentence_transformers import util

def all_lines(path):
  content = ''
  # ignore emoji
  with open(path, encoding='utf-8', errors='ignore') as f:
    content = f.readlines()
  return content

def get_files(path):
  return [ join(path, f) for f in listdir(path) if isfile(join(path, f))]

  
def search(query, corpus):
  hits = util.semantic_search(query, corpus, top_k=3, score_function=util.dot_score)
  return hits[0]


def flatten(l):
  return  [item for sublist in l for item in sublist]