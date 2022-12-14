from .libs import flatten
import math

class SentenceChunk:
  @staticmethod
  def chunk(document: str) -> 'list[str]':
    paragraphs = document.split('\n')
    sentences = flatten([paragraph.split('.') for paragraph in paragraphs])
    cleaned = list(map(lambda x: x.strip(), sentences))
    return list(filter(lambda x: x!='', cleaned))
  

class ParagraphChunk:
  @staticmethod
  def chunk(document: str) -> 'list[str]':
    paragraphs = document.split('\n')
    cleaned = list(map(lambda x:x.strip(), paragraphs))
    return list(filter(lambda x: x!='', cleaned))


class Chunkless:
  @staticmethod
  def chunk(document: str) -> 'list[str]':
    return [document]


class NthParagraphChunk:
  @staticmethod
  def chunk(document:str, count:int) -> 'list[str]':
    paragraphs = ParagraphChunk.chunk(document)
    steps = math.ceil(len(paragraphs) / count)
    return ["\n".join(paragraphs[i:i + steps]) for i in range(0, len(paragraphs), steps)]


class NthChunk:
  @staticmethod
  def chunk(document:str, count:int) -> 'list[str]':
    chunks = ParagraphChunk.chunk(document) if '\n' in document else SentenceChunk.chunk(document)
    steps = math.ceil(len(chunks) / count)
    return ["\n".join(chunks[i:i + steps]) for i in range(0, len(chunks), steps)]

