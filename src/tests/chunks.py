from ..libs import expect, get_content
from ..chunks import Chunkless, SentenceChunk, ParagraphChunk


def run_test():
  content = get_content('doc/chunking-sample.txt')
  
  chunkless = Chunkless.chunk(content)
  expect(len(chunkless) == 1, 'Chunkless chunk produce a single chunk')

  paragraphChunk = ParagraphChunk.chunk(content)
  expect(len(paragraphChunk) == 3, 'Paragraph chunk produce three chunks')
  
  sentenceChunk = SentenceChunk.chunk(content)
  expect(len(sentenceChunk) == 5, 'Sentence chunk produce five chunks')


if __name__ == '__main__':
  run_test()