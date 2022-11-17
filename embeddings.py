from sentence_transformers import SentenceTransformer
from libs import all_lines, flatten
import torch
import numpy

model = SentenceTransformer('msmarco-distilbert-base-tas-b')

class AbstractEmbeddingEncoder():
  @staticmethod
  def document(chunk: 'list[str]') -> torch.Tensor:
    raise NotImplementedError("trying to call method of an abstract class")

  @staticmethod
  def query(query: str) -> torch.Tensor:
    raise NotImplementedError("trying to call method of an abstract class")

# This strategy treat the entire document as a string of maximum 512 word
class LongSentence512(AbstractEmbeddingEncoder): 

  @staticmethod
  def document(chunks: 'list[str]') -> torch.Tensor:
    blob = '\n'.join(chunks)
    embedding = model.encode(blob, convert_to_tensor=True)
    return embedding

  @staticmethod
  def query(query: str) -> torch.Tensor:
    return model.encode(query, convert_to_tensor=True)


class Mean(AbstractEmbeddingEncoder):

  @staticmethod
  def document(chunks: 'list[str]') -> torch.Tensor:
    embeddings = model.encode(chunks)
    embedding = numpy.mean(embeddings, axis=0)
    return torch.tensor(embedding)  

  @staticmethod
  def query(query: str) -> torch.Tensor:
    return model.encode(query, convert_to_tensor=True)


class AMaxParagraph(AbstractEmbeddingEncoder):

  @staticmethod
  def document(chunks: 'list[str]') -> torch.Tensor:
    embeddings = model.encode(chunks)
    embedding = numpy.amax(embeddings, axis=0)
    return torch.tensor(embedding)

  @staticmethod
  def query(query: str) -> torch.Tensor:
    return model.encode(query, convert_to_tensor=True)


class SumAMax(AbstractEmbeddingEncoder):
  @staticmethod
  def calc_unit(s: numpy.ndarray) -> torch.Tensor:
    mean = numpy.mean(s, axis=0)
    amax = numpy.amax(s, axis=0)
    concatted = numpy.concatenate([mean, amax])
    unit = concatted / (concatted**2).sum()**0.5
    return torch.tensor(unit)

  @staticmethod
  def document(chunks: 'list[str]') -> torch.Tensor:
    embeddings = model.encode(chunks)
    return SumAMax.calc_unit(embeddings)

  @staticmethod
  def query(query: str) -> torch.Tensor:
    embedding = model.encode(query)
    return SumAMax.calc_unit(embedding)