from sentence_transformers import SentenceTransformer, util
import numpy
from ..libs import expect
from ..sentence_transformer import SentenceTransformerSpecb

def encode(model, corpus):
  return model.encode(corpus, batch_size=1)

def verify_encode_batch_tensor_equality(model):
  value = "Hello world. I am the same string every where. Let's see how my embedding change."
  corpus = [value]
  batch0 = encode(model, corpus)
  batch1 = encode(model, corpus)

  # single = model.encode([value])
  
  corpus.append("Now I am lengthened")
  batch2 = encode(model, corpus)

  expect(numpy.array_equal(batch1[0], batch0[0]), 'Expect embedding of the same value does not change between batch0 and batch 1')

  expect(numpy.array_equal(batch1[0], batch2[0]), 'Expect embedding of the same value does not change between batch')
  # https://github.com/UKPLab/sentence-transformers/issues/1729
  expect(not numpy.array_equal(batch2[0], batch2[1]), 'Expect embedding of different value is different')

  corpus.append(value)
  batch3 = encode(model, corpus)

  expect(numpy.array_equal(batch3[0], batch3[2]), 'Expect embeddings of same value is equal')
  # https://github.com/UKPLab/sentence-transformers/issues/1729
  expect(numpy.array_equal(batch1[0], batch3[0]), 'Expect embeddings of same value from first batch and third batch is equal')

  corpus.append(value)
  batch4 = encode(model, corpus)

  expect(numpy.array_equal(batch4[0], batch4[3]), 'Expect embeddings of same value is equal')
  expect(numpy.array_equal(batch1[0], batch3[0]), 'Expect embeddings of same value from first batch and third batch is equal')
  expect(numpy.array_equal(batch3[0], batch4[0]), 'Expect embeddings of same value from third batch and third batch is equal')


def verify_encode_batch_dot_product(model):
  value = "Hello world. I am the same string every where. Let's see how my embedding change."
  corpus = [value]
  batch0 = model.encode(corpus, convert_to_tensor=True)
  batch1 = model.encode(corpus, convert_to_tensor=True)

  # single = model.encode([value])
  
  corpus.append("Now I am lengthened")
  batch2 = model.encode(corpus, convert_to_tensor=True)

  corpus.append(value)
  batch3 = model.encode(corpus, convert_to_tensor=True)


  product =[ 
    util.dot_score(batch0[0], batch1[0]),
    util.dot_score(batch1[0], batch2[0]),
    util.dot_score(batch2[0], batch3[0]),
    util.dot_score(batch2[0], batch3[2]),
  ]
  print(product)


def verify_encode_max_sequence_length_behaviour(model):
  sentence_512 = "the region of Transoxiana had been conquered by the Muslim Arabs of the Syria-based Umayyad Caliphate under Qutayba ibn Muslim in the reign of al-Walid I (r. 705–715), following the Muslim conquest of Persia and of Khurasan in the mid-7th century.[1] The loyalties of the region's native Iranian and Turkic inhabitants and autonomous local rulers remained volatile, and in 719, they sent a petition to the Chinese and their vassals the Türgesh (a Turkic tribal confederation) for military aid against the Muslims.[2] In response, Türgesh attacks began in 720, and the native Sogdians launched uprisings against the Caliphate. These were suppressed with great brutality by the governor of Khurasan, Sa'id ibn Amr al-Harashi, but in 724 his successor, Muslim ibn Sa'id al-Kilabi, suffered a major disaster (the so-called Day of Thirst) while trying to capture Ferghana.[3][4] For the next few years, Umayyad forces were limited to the defensive. Efforts to placate and win the support of the local population by abolishing taxation of the native converts to Islam (mawali) were undertaken, but these were half-hearted and soon reversed, while heavy-handed Arab actions further alienated the local elites. In 728 a large-scale uprising, coupled with a Türgesh invasion, led to the abandonment of most of Transoxiana by the Caliphate's forces, except for the region around Samarkand.[5][6]In the hope of reversing the situation, in early 730 Caliph Hisham ibn Abd al-Malik (r. 723–743) appointed a new governor in Khurasan: the experienced general Junayd ibn Abd al-Rahman al-Murri, who had been recently engaged in the pacification of Sindh. The difficult security situation at the time is illustrated by the fact that Junayd needed an escort of 7,000 cavalry after crossing the Oxus River, and that he was attacked by the Türgesh khagan Suluk while riding to link up with the army of his predecessor, Ashras al-Sulami, who in the previous year had advanced up to Bukhara in a hard-fought campaign. After difficult fighting, Junayd and his escort were able to repel the attack and link up with al-Sulami's forces. Bukhara and most of Sogdiana was recovered soon after, as the Türgesh army withdrew north towards Samarkand. The Muslim army followed and scored a victory in a battle fought near the city. Junayd then retired with his troops to winter in Merv.[7][8] During the winter, rebellions broke out south of the Oxus in Tokharistan, which had previously been quiescent under Muslim rule. Junayd was forced to set out for Balkh and there dispersed 28,000 of his men to quell the revolt. This left him seriously short of men when, in early 731, the Türgesh laid siege to Samarkand and appeals for aid arrived from the city's governor, Sawra ibn al-Hurr al-Abani. Despite the opinion of the army's veteran Khurasani Arab leaders, who counselled that he should wait to reassemble his forces and not cross the Oxus with less than 50,000 men Junayd resolved to march immediately to Samarkand's rescue. This left him seriously short of men when, in early 731, the Türgesh laid siege to Samarkand and appeals for aid arrived from the city"
  sentence_513 = "the region of Transoxiana had been conquered by the Muslim Arabs of the Syria-based Umayyad Caliphate under Qutayba ibn Muslim in the reign of al-Walid I (r. 705–715), following the Muslim conquest of Persia and of Khurasan in the mid-7th century.[1] The loyalties of the region's native Iranian and Turkic inhabitants and autonomous local rulers remained volatile, and in 719, they sent a petition to the Chinese and their vassals the Türgesh (a Turkic tribal confederation) for military aid against the Muslims.[2] In response, Türgesh attacks began in 720, and the native Sogdians launched uprisings against the Caliphate. These were suppressed with great brutality by the governor of Khurasan, Sa'id ibn Amr al-Harashi, but in 724 his successor, Muslim ibn Sa'id al-Kilabi, suffered a major disaster (the so-called Day of Thirst) while trying to capture Ferghana.[3][4] For the next few years, Umayyad forces were limited to the defensive. Efforts to placate and win the support of the local population by abolishing taxation of the native converts to Islam (mawali) were undertaken, but these were half-hearted and soon reversed, while heavy-handed Arab actions further alienated the local elites. In 728 a large-scale uprising, coupled with a Türgesh invasion, led to the abandonment of most of Transoxiana by the Caliphate's forces, except for the region around Samarkand.[5][6]In the hope of reversing the situation, in early 730 Caliph Hisham ibn Abd al-Malik (r. 723–743) appointed a new governor in Khurasan: the experienced general Junayd ibn Abd al-Rahman al-Murri, who had been recently engaged in the pacification of Sindh. The difficult security situation at the time is illustrated by the fact that Junayd needed an escort of 7,000 cavalry after crossing the Oxus River, and that he was attacked by the Türgesh khagan Suluk while riding to link up with the army of his predecessor, Ashras al-Sulami, who in the previous year had advanced up to Bukhara in a hard-fought campaign. After difficult fighting, Junayd and his escort were able to repel the attack and link up with al-Sulami's forces. Bukhara and most of Sogdiana was recovered soon after, as the Türgesh army withdrew north towards Samarkand. The Muslim army followed and scored a victory in a battle fought near the city. Junayd then retired with his troops to winter in Merv.[7][8] During the winter, rebellions broke out south of the Oxus in Tokharistan, which had previously been quiescent under Muslim rule. Junayd was forced to set out for Balkh and there dispersed 28,000 of his men to quell the revolt. This left him seriously short of men when, in early 731, the Türgesh laid siege to Samarkand and appeals for aid arrived from the city's governor, Sawra ibn al-Hurr al-Abani. Despite the opinion of the army's veteran Khurasani Arab leaders, who counselled that he should wait to reassemble his forces and not cross the Oxus with less than 50,000 men Junayd resolved to march immediately to Samarkand's rescue. This left him seriously short of men when, in early 731, the Türgesh laid siege to Samarkand and appeals for aid arrived from the city's gorvernor."
  
  embedding_512 = model.encode(sentence_512)
  embedding_513 = model.encode(sentence_513)
  
  expect(model.max_seq_length == 512, 'Expected model.max_sequence_length is equal to 512')
  expect(numpy.array_equal(embedding_512, embedding_513), 'Expected embedding of 512 words equal to embedding of 513 word')

  batch = model.encode([sentence_512, "hello world", sentence_513])

  expect(numpy.array_equal(batch[0], batch[2]), 'Expected first item equal third item in batch')
  expect(not numpy.array_equal(batch[0], batch[1]), 'Expected first item not equal to second item in batch')
  

def verify_normalization_behaviour(model:SentenceTransformer):
  content = "What type of cat is suitable for cold climate?"
  embedding = model.encode([content])
  embedding_normalize = model.encode([content], normalize_embeddings=True)
 
  documents = [
    'The house cat enjoy favourable living space where temperature is more constant. This means the evolve to enjoy more moderate climate.',
    'The dessert cat adapted to the hot climate and generally very short fur. There short fur also means that you do not have to clean up as often.',
    'The artic cat has very long fur and more body fat than any other type of cat. This means that they can live in colder area of the earth.'
  ]

  document_embeddings_standard = model.encode(documents)
  document_embeddings_normalized = model.encode(documents)

  standard_score = [ util.dot_score(embedding, doc_embedding) for doc_embedding in document_embeddings_standard]
  normalize_score = [ util.dot_score(embedding_normalize, doc_embedding) for doc_embedding in document_embeddings_normalized]

  print("standard score", standard_score)
  print("normalised score", normalize_score)

  expect(len(embedding) == 1, "Expect embedding is 1. Receive %d" % len(embedding))

if __name__ == '__main__':
  model = SentenceTransformer('msmarco-distilbert-base-tas-b')
  # verify_encode_max_sequence_length_behaviour(model)
  # print('Distilled Bert')
  # verify_encode_batch_tensor_equality(model)
  # print('SGPT')
  # verify_encode_batch_tensor_equality(mmodelSpecb)
  # verify_encode_batch_dot_product(model)
  verify_normalization_behaviour(model)
