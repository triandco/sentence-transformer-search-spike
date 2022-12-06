# Sentence embedding using nth block embeddings
Nth block embeddings divide a document into n blocks. Each block received its own embedding calculated using numpy.Mean and numpy.Amax of each paragraph inside the block.
A document's embeddings compared to query embedding using dot product for the highest score.

Dimensions of the embedding is reduced using [🤗 sentence embeddings guide](https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/distillation/dimensionality_reduction.py)

# msmarco-bert-base-dot-v5 with dot product score
| Number of block | Dimension | Tests passed | Doc Embedding generation time | Embedding size kb |
|-----------------|-----------|--------------|-------------------------------|-------------------|
| 1 | 768 | 4/8 | 18.176465s | 6.144 |
| 2 | 768 | 3/8 | 15.738416s | 12.288 |
| 2 | 128 | 2/8 | 15.967346s | 2.048 |
| 3 | 128 | 2/8 | 18.136934s | 3.072 |
| 6 | 128 | 4/8 | 22.678251s | 6.144 |
| 2 | 265 | 1/8 | 15.889265s | 4.24 |
| 3 | 256 | 3/8 | 18.072454s | 6.144  |
| 4 | 256 | 3/8 | 19.917356s | 8.192 |
| 2 | 384 | 5/8 | 37.398320s | 6.144 |
| 3 | 384 | 3/8 | 50.008942s | 9.216 |
| 1 | 512 | 3/8 | 77.287130s | 4.096  |
| 2 | 512 | 3/8 | 96.657237s | 8.192 |


# msmarco-distilbert-base-tas-b with dot product score
| Number of block | Dimension | Tests passed | Doc Embedding generation time | Embedding size kb |
|-----------------|-----------|--------------|-------------------------------|-------------------|
| 1 | 768 | 5/8 | 20.487066s | 6.144 |
| 2 | 768 | 4/8 | 17.733734s | 12.288 |
| 2 | 128 | 3/8 | 18.020780s | 2.048 |
| 3 | 128 | 4/8 | 20.452163s | 3.072 |
| 6 | 128 | 3/8 | 25.586701s | 6.144 |
| 2 | 256 | 3/8 | 17.925052s | 4.24 |
| 3 | 256 | 3/8 | 16.339944s | 6.144 |
| 4 | 256 | 3/8 | 19.347116s | 8.192 |
| 2 | 384 | 4/8 | 18.317890s | 6.144 |
| 3 | 384 | 4/8 | 17.110831s | 9.216 |
| 1 | 512 | 3/8 | 12.436955s | 4.096 |
| 2 | 512 | 3/8 | 14.385524s | 8.192 |


# msmarco-distilbert-cos-v5 with cosine similarity score
| Number of block | Dimension | Tests passed | Doc Embedding generation time | Embedding size kb |
|-----------------|-----------|--------------|-------------------------------|-------------------|
| 1 | 768 | 3/8 | 17.374597s | 6.144 |
| 2 | 768 | 5/8 | 18.302987s | 12.288 |
| 2 | 128 | 3/8 | 18.498830s | 2.048 |
| 3 | 128 | 3/8 | 17.567860s | 3.072 |
| 6 | 128 | 3/8 | 22.061858s | 6.144 |
| 2 | 256 | 4/8 | 16.189130s | 4.24 |
| 3 | 256 | 4/8 | 15.234358s | 6.144 |
| 4 | 256 | 3/8 | 22.863533s | 8.192 |
| 2 | 384 | 4/8 | 13.072537s | 6.144 |
| 3 | 384 | 4/8 | 18.567012s | 9.216 |
| 1 | 512 | 3/8 | 11.810111s | 4.096 |
| 2 | 512 | 3/8 | 10.768994s | 8.192 |
