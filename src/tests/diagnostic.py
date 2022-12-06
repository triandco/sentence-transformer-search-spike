from ..libs import get_files, get_content, flatten
from sentence_transformers import SentenceTransformer


def print_embedding_size(model_name):
  model = SentenceTransformer(model_name)
  print('---------------------------')
  print('model:', model_name)
  example_embedding = model.encode("In this cat humble opinion, caviar and lobster has become way too extravagant for dinner as oppose to a simple but delightful himono from the far orient.")
  print("Standard embedding length", len(example_embedding))
  embedding_size = example_embedding[0].itemsize * len(example_embedding)
  print("Single dimension size kb: ", example_embedding[0].itemsize)
  print("Standard embedding size in kb: ", embedding_size / 1000)


if __name__=='__main__':
  directory_path = 'doc/not-boring-podcast'
  files = get_files(directory_path)
  documents = list(zip(files, list(map(get_content, files))))
  paragraphs = [ document[1].split("\n") for document in documents ]
  
  print("Number of documents:", len(documents))
  print("Number of paragraph:", len(flatten(paragraphs)))

  model_names = [
    'msmarco-distilbert-base-tas-b',
    'models/reduced-128/msmarco-distilbert-cos-v5',
    'models/reduced-256/msmarco-bert-base-dot-v5'
  ]
  for model_name in model_names:
    print_embedding_size(model_name)

  
