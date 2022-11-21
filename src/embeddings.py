from sentence_transformers import SentenceTransformer, util
from .sentence_transformer import SentenceTransformerSpecb
from .chunks import Chunkless, ParagraphChunk
from scipy.spatial.distance import cosine
import torch
import numpy


class AbstractEmbeddingEncoder():
  def document(document: str) -> torch.Tensor:
    raise NotImplementedError("trying to call method of an abstract class")

  def query(query: str) -> torch.Tensor:
    raise NotImplementedError("trying to call method of an abstract class")

  def score(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: 
    raise NotImplemented("trying to call method of an abstract class")

  def chunk(document: str) -> 'list[str]':
    raise NotImplemented("trying to call a method of an abstract class")


class DotProductScore():
  @staticmethod
  def score(a: torch.Tensor, b: torch.Tensor): 
    return util.dot_score(a, b)


class CosineSimiliarityScore():
  @staticmethod
  def score(a: torch.Tensor, b: torch.Tensor):
    return util.cos_sim(a,b)


# This strategy treat the entire document as a string of maximum 512 word
class WholeChunk(DotProductScore, Chunkless, AbstractEmbeddingEncoder): 

  def __init__(self, model: SentenceTransformer):
    super().__init__()
    self.model = model

  def document(self, document: str) -> torch.Tensor:
    chunks = WholeChunk.chunk(document)
    embedding = self.model.encode(chunks, convert_to_tensor=True)
    return embedding[0]

  def query(self, query: str) -> torch.Tensor:
    return self.model.encode(query, convert_to_tensor=True)



# This strategy treat the entire document as multiple sentence of maximum 512 word
class Mean(DotProductScore, ParagraphChunk, AbstractEmbeddingEncoder):
  
  def __init__(self, model: SentenceTransformer): 
    super().__init__()
    self.model = model

  def document(self, document: str) -> torch.Tensor:
    chunks = Mean.chunk(document)
    embeddings = self.model.encode(chunks)
    embedding = numpy.mean(embeddings, axis=0)
    return torch.tensor(embedding, device='cuda:0')  

  def query(self, query: str) -> torch.Tensor:
    return self.model.encode(query, convert_to_tensor=True)


class AMax(DotProductScore, ParagraphChunk, AbstractEmbeddingEncoder):
  def __init__(self, model: SentenceTransformer):
    super().__init__()
    self.model = model


  def document(self, document: str) -> torch.Tensor:
    chunks = AMax.chunk(document)
    embeddings = self.model.encode(chunks)
    embedding = numpy.amax(embeddings, axis=0)
    return torch.tensor(embedding, device='cuda:0')


  def query(self, query: str) -> torch.Tensor:
    return self.model.encode(query, convert_to_tensor=True)

#https://dev.to/mage_ai/how-to-build-a-search-engine-with-word-embeddings-56jd
class MeanAMax(ParagraphChunk, DotProductScore, AbstractEmbeddingEncoder):
  def __init__(self, model: SentenceTransformer):
    super().__init__()
    self.model = model


  def document(self, document: str) -> torch.Tensor:
    chunks = MeanAMax.chunk(document)
    embeddings = self.model.encode(chunks)
    mean = numpy.mean(embeddings, axis=0)
    amax = numpy.amax(embeddings, axis=0)
    concatted = numpy.concatenate([mean, amax])
    # unit = concatted / (concatted**2).sum()**0.5
    return torch.tensor(concatted, device='cuda:0')


  def query(self, query: str) -> torch.Tensor:
    embedding = self.model.encode(query)
    concatted = numpy.concatenate([embedding, embedding])
    # unit = concatted / (concatted**2).sum()**0.5
    return torch.tensor(concatted, device='cuda:0')


class SGPTCosine(ParagraphChunk, CosineSimiliarityScore, AbstractEmbeddingEncoder): 
  def __init__(self, model: SentenceTransformerSpecb):
    super().__init__()
    self.model = model

  
  def document(self, document: str) -> torch.Tensor:
    chunks = MeanAMax.chunk(document)
    embeddings = self.model.encode(chunks)
    mean = numpy.mean(embeddings, axis=0)
    amax = numpy.amax(embeddings, axis=0)
    concatted = numpy.concatenate([mean, amax])
    # unit = concatted / (concatted**2).sum()**0.5
    return torch.tensor(concatted, device='cuda:0')


  def query(self, query: str) -> torch.Tensor:
    embedding = self.model.encode(query)
    concatted = numpy.concatenate([embedding, embedding])
    # unit = concatted / (concatted**2).sum()**0.5
    return torch.tensor(concatted, device='cuda:0')