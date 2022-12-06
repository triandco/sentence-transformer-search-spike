"""
Copy from https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/distillation/dimensionality_reduction.py
"""
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, LoggingHandler, util, evaluation, models, InputExample
import logging
import os
import gzip
import csv
import random
import numpy as np
import torch


def reduce_dimension(model_name, new_dimension=128):
    model = SentenceTransformer(model_name)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    logger = logging.getLogger(__name__)

    nli_dataset_path = 'datasets/AllNLI.tsv.gz'
    sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

    if not os.path.exists(nli_dataset_path):
        util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

    if not os.path.exists(sts_dataset_path):
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)


    logger.info("Read STSbenchmark test dataset")
    eval_examples = []
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'test':
                score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
                eval_examples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

    # Evaluate the original model on the STS benchmark dataset
    stsb_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(eval_examples, name='sts-benchmark-test')

    logger.info("Original model performance:")
    stsb_evaluator(model)

    nli_sentences = set()
    with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            nli_sentences.add(row['sentence1'])
            nli_sentences.add(row['sentence2'])

    nli_sentences = list(nli_sentences)
    random.shuffle(nli_sentences)

    pca_train_sentences = nli_sentences[0:20000]
    train_embeddings = model.encode(pca_train_sentences, convert_to_numpy=True)

    pca = PCA(n_components=new_dimension)
    pca.fit(train_embeddings)
    pca_comp = np.asarray(pca.components_)

    dense = models.Dense(in_features=model.get_sentence_embedding_dimension(), out_features=new_dimension, bias=False, activation_function=torch.nn.Identity())
    dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
    model.add_module('dense', dense)

    # Evaluate the model with the reduce embedding size
    logger.info("Model with {} dimensions:".format(new_dimension))
    stsb_evaluator(model)

    file_name = 'models/reduced-%s/%s' % (new_dimension, model_name)
    model.save(file_name)

    return file_name

if __name__=='__main__':
    model_names = [
        'msmarco-distilbert-base-tas-b', 
        'sentence-transformers/msmarco-distilbert-cos-v5', 
        'sentence-transformers/msmarco-bert-base-dot-v5'
    ] 
    varieties = [ 512 ]
    for variety in varieties:
      outcome = [ reduce_dimension(model, variety) for model in model_names]
    