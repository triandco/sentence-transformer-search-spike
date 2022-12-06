from sentence_transformers import SentenceTransformer, util
from .sentence_transformer import SentenceTransformerSpecb
from .chunks import Chunkless, ParagraphChunk, NthChunk
from scipy.spatial.distance import cosine
import torch
import numpy


class AbstractQueryEncoder():
  def query(query: str) -> torch.Tensor:
    raise NotImplementedError("trying to call method of an abstract class")


class AbstractEmbeddingStrategy(AbstractQueryEncoder):
  def __init__(self, model: SentenceTransformer, model_name: str) -> None:
    super().__init__()
    self.model = model
    self.model_name = model_name

  def __str__(self) -> str:
    return "%s %s" % (self.__class__.__name__, self.model_name)

  def document(document:str):
    raise NotImplementedError("trying to call method of an abstract class")

  def score(self, query: torch.Tensor, doc) -> torch.Tensor: 
    raise NotImplemented("trying to call method of an abstract class")


class AbstractDocumentEncoder():
  def document(document: str) -> torch.Tensor:
    raise NotImplementedError("trying to call method of an abstract class")
  
  def score(query: torch.Tensor, doc: torch.Tensor) -> torch.Tensor: 
    raise NotImplemented("trying to call method of an abstract class")


class AbstractCollectionEncoder():
  def document(document: str) -> 'list[torch.Tensor]':
    raise NotImplementedError("trying to call method of an abstract class")

  def score(query: torch.Tensor, doc: 'list[torch.Tensor]') -> torch.Tensor: 
    raise NotImplemented("trying to call method of an abstract class")


class DotProductCollectionScore():
  def score(self, query: torch.Tensor, doc: 'list[torch.Tensor]') -> torch.Tensor:
    scores = [util.dot_score(query, chunk) for chunk in doc]
    return sorted(scores, reverse=True)[0]


class CosineSimilarityCollectionScore():
  def score(self, query: torch.Tensor, doc: 'list[torch.Tensor]') -> torch.Tensor:
    scores = [util.cos_sim(query, chunk) for chunk in doc]
    return sorted(scores, reverse=True)[0]


class DotProductScore():
  def score(self, a: torch.Tensor, b: torch.Tensor): 
    return util.dot_score(a, b)


class CosineSimiliarityScore():
  def score(self, a: torch.Tensor, b: torch.Tensor):
    return util.cos_sim(a,b)


# This strategy treat the entire document as a string of maximum 512 word
class WholeChunk(DotProductScore, Chunkless, AbstractEmbeddingStrategy, AbstractDocumentEncoder): 

  def document(self, document: str) -> torch.Tensor:
    chunks = WholeChunk.chunk(document)
    embedding = self.model.encode(chunks, convert_to_tensor=True)
    return embedding[0]

  def query(self, query: str) -> torch.Tensor:
    return self.model.encode(query, convert_to_tensor=True)


# This strategy treat the entire document as multiple sentence of maximum 512 word
class Mean(DotProductScore, ParagraphChunk, AbstractEmbeddingStrategy, AbstractDocumentEncoder):
  
  def document(self, document: str) -> torch.Tensor:
    chunks = Mean.chunk(document)
    embeddings = self.model.encode(chunks)
    embedding = numpy.mean(embeddings, axis=0)
    return torch.tensor(embedding, device='cuda:0')  

  def query(self, query: str) -> torch.Tensor:
    return self.model.encode(query, convert_to_tensor=True)


class AMax(DotProductScore, ParagraphChunk, AbstractEmbeddingStrategy, AbstractDocumentEncoder):

  def document(self, document: str) -> torch.Tensor:
    chunks = AMax.chunk(document)
    embeddings = self.model.encode(chunks)
    embedding = numpy.amax(embeddings, axis=0)
    return torch.tensor(embedding, device='cuda:0')


  def query(self, query: str) -> torch.Tensor:
    return self.model.encode(query, convert_to_tensor=True)

#https://dev.to/mage_ai/how-to-build-a-search-engine-with-word-embeddings-56jd
class WholeDocument(ParagraphChunk, DotProductScore, AbstractEmbeddingStrategy, AbstractDocumentEncoder):

  def document(self, document: str) -> torch.Tensor:
    chunks = WholeDocument.chunk(document)
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


class NthBlock(NthChunk, AbstractEmbeddingStrategy, AbstractDocumentEncoder):
  def __init__(self, model: SentenceTransformer, model_name:str, count: int):
    super().__init__(model, model_name)
    self.count = count

  def document(self, document: str) -> 'list[torch.Tensor]':
    chunks = NthBlock.chunk(document, self.count)
    
    tensors = []
    for chunk in chunks:
      paragraphs = chunk.split("\n")
      embeddings = self.model.encode(paragraphs)
      mean = numpy.mean(embeddings, axis=0)
      amax = numpy.amax(embeddings, axis=0)
      concatted = numpy.concatenate([mean, amax])
      tensor = torch.tensor(concatted, device='cuda:0')
      tensors.append(tensor)
    
    return tensors

  def query(self, query: str) -> torch.Tensor:
    embedding = self.model.encode(query)
    concatted = numpy.concatenate([embedding, embedding])
    return torch.tensor(concatted, device='cuda:0')


class NthBlockDot(DotProductCollectionScore, NthBlock):
  def __str__(self) -> str:
    return "NthBlockDot with %d block using %s" % (self.count, self.model_name)


class NthBlockCosine(CosineSimilarityCollectionScore, NthBlock):
  def __str__(self) -> str:
    return "NthBlockCosine with %d block using %s" % (self.count, self.model_name)


class SGPTCosine(ParagraphChunk, CosineSimiliarityScore, AbstractEmbeddingStrategy, AbstractDocumentEncoder): 
  def __init__(self, model: SentenceTransformerSpecb, model_name:str):
    super().__init__()
    self.model = model
    self.model_name = model_name

  
  def document(self, document: str) -> torch.Tensor:
    chunks = SGPTCosine.chunk(document)
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


class ParagraphBlock(ParagraphChunk, DotProductCollectionScore, AbstractEmbeddingStrategy, AbstractCollectionEncoder):
  def document(self, document:str) -> 'list[torch.Tensor]':
    chunks = ParagraphBlock.chunk(document)
    return self.model.encode(chunks)


  def query(self, query:str) -> torch.Tensor:
    return self.model.encode(query)

