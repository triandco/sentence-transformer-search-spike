from .libs import flatten

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


class NthChunk:
  @staticmethod
  def chunk(document:str, count) -> 'list[str]':
    paragraphs = document.split('\n')
    return ["\n".join(paragraphs[i:i + count]) for i in range(0, len(paragraphs), count)]