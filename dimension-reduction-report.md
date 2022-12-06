# Evaluating reduced dimension embeddings performance 

2022-12-05 19:32:05 - Read STSbenchmark test dataset
██████████████████████████████████████████████████████████████████████████████
2022-12-05 19:32:05 - Original model performance:
2022-12-05 19:32:05 - EmbeddingSimilarityEvaluator: Evaluating the model on sts-benchmark-test dataset:
2022-12-05 19:32:10 - Cosine-Similarity :       Pearson: 0.7913 Spearman: 0.7839
2022-12-05 19:32:10 - Manhattan-Distance:       Pearson: 0.7880 Spearman: 0.7789
2022-12-05 19:32:10 - Euclidean-Distance:       Pearson: 0.7886 Spearman: 0.7794
2022-12-05 19:32:10 - Dot-Product-Similarity:   Pearson: 0.7671 Spearman: 0.7501
Batches: 100%|| 625/625 [00:37<00:00, 16.73it/s]
2022-12-05 19:32:55 - Model with 128 dimensions:
2022-12-05 19:32:55 - EmbeddingSimilarityEvaluator: Evaluating the model on sts-benchmark-test dataset:
2022-12-05 19:33:00 - Cosine-Similarity :       Pearson: 0.7841 Spearman: 0.7771
2022-12-05 19:33:00 - Manhattan-Distance:       Pearson: 0.7678 Spearman: 0.7526
2022-12-05 19:33:00 - Euclidean-Distance:       Pearson: 0.7643 Spearman: 0.7506
2022-12-05 19:33:00 - Dot-Product-Similarity:   Pearson: 0.5874 Spearman: 0.5666
2022-12-05 19:33:00 - Save model to models/reduced-128/msmarco-distilbert-base-tas-b
██████████████████████████████████████████████████████████████████████████████
2022-12-05 19:33:01 - Load pretrained SentenceTransformer: sentence-transformers/msmarco-distilbert-cos-v5
2022-12-05 19:33:02 - Use pytorch device: cuda
2022-12-05 19:33:02 - Read STSbenchmark test dataset
2022-12-05 19:33:03 - Original model performance:
2022-12-05 19:33:03 - EmbeddingSimilarityEvaluator: Evaluating the model on sts-benchmark-test dataset:
2022-12-05 19:33:08 - Cosine-Similarity :       Pearson: 0.7950 Spearman: 0.7824
2022-12-05 19:33:08 - Manhattan-Distance:       Pearson: 0.7854 Spearman: 0.7821
2022-12-05 19:33:08 - Euclidean-Distance:       Pearson: 0.7857 Spearman: 0.7824
2022-12-05 19:33:08 - Dot-Product-Similarity:   Pearson: 0.7950 Spearman: 0.7824
Batches: 100%|| 625/625 [00:37<00:00, 16.61it/s]
2022-12-05 19:33:53 - Model with 128 dimensions:
2022-12-05 19:33:53 - EmbeddingSimilarityEvaluator: Evaluating the model on sts-benchmark-test dataset:
2022-12-05 19:33:59 - Cosine-Similarity :       Pearson: 0.7898 Spearman: 0.7803
2022-12-05 19:33:59 - Manhattan-Distance:       Pearson: 0.7800 Spearman: 0.7737
2022-12-05 19:33:59 - Euclidean-Distance:       Pearson: 0.7885 Spearman: 0.7798
2022-12-05 19:33:59 - Dot-Product-Similarity:   Pearson: 0.7334 Spearman: 0.7146
2022-12-05 19:33:59 - Save model to models/reduced-128/sentence-transformers/msmarco-distilbert-cos-v5
██████████████████████████████████████████████████████████████████████████████
2022-12-05 19:34:00 - Load pretrained SentenceTransformer: sentence-transformers/msmarco-bert-base-dot-v5
2022-12-05 19:34:02 - Use pytorch device: cuda
2022-12-05 19:34:02 - Read STSbenchmark test dataset
2022-12-05 19:34:02 - Original model performance:
2022-12-05 19:34:02 - EmbeddingSimilarityEvaluator: Evaluating the model on sts-benchmark-test dataset:
2022-12-05 19:34:13 - Cosine-Similarity :       Pearson: 0.7682 Spearman: 0.7607
2022-12-05 19:34:13 - Manhattan-Distance:       Pearson: 0.7601 Spearman: 0.7539
2022-12-05 19:34:13 - Euclidean-Distance:       Pearson: 0.7608 Spearman: 0.7546
2022-12-05 19:34:13 - Dot-Product-Similarity:   Pearson: 0.6881 Spearman: 0.6961
Batches: 100%|| 625/625 [01:13<00:00,  8.50it/s]
2022-12-05 19:35:34 - Model with 128 dimensions:
2022-12-05 19:35:34 - EmbeddingSimilarityEvaluator: Evaluating the model on sts-benchmark-test dataset:
2022-12-05 19:35:44 - Cosine-Similarity :       Pearson: 0.7744 Spearman: 0.7638
2022-12-05 19:35:44 - Manhattan-Distance:       Pearson: 0.7549 Spearman: 0.7421
2022-12-05 19:35:44 - Euclidean-Distance:       Pearson: 0.7605 Spearman: 0.7471
2022-12-05 19:35:44 - Dot-Product-Similarity:   Pearson: 0.6041 Spearman: 0.5814
2022-12-05 19:35:44 - Save model to models/reduced-128/sentence-transformers/msmarco-bert-base-dot-v5
██████████████████████████████████████████████████████████████████████████████
2022-12-05 19:35:45 - Load pretrained SentenceTransformer: msmarco-distilbert-base-tas-b
2022-12-05 19:35:46 - Use pytorch device: cuda
2022-12-05 19:35:46 - Read STSbenchmark test dataset
2022-12-05 19:35:46 - Original model performance:
2022-12-05 19:35:46 - EmbeddingSimilarityEvaluator: Evaluating the model on sts-benchmark-test dataset:
2022-12-05 19:35:52 - Cosine-Similarity :       Pearson: 0.7913 Spearman: 0.7839
2022-12-05 19:35:52 - Manhattan-Distance:       Pearson: 0.7880 Spearman: 0.7789
2022-12-05 19:35:52 - Euclidean-Distance:       Pearson: 0.7886 Spearman: 0.7794
2022-12-05 19:35:52 - Dot-Product-Similarity:   Pearson: 0.7671 Spearman: 0.7501
Batches: 100%|| 625/625 [00:37<00:00, 16.77it/s]
2022-12-05 19:36:37 - Model with 256 dimensions:
2022-12-05 19:36:37 - EmbeddingSimilarityEvaluator: Evaluating the model on sts-benchmark-test dataset:
2022-12-05 19:36:42 - Cosine-Similarity :       Pearson: 0.7975 Spearman: 0.7879
2022-12-05 19:36:42 - Manhattan-Distance:       Pearson: 0.7756 Spearman: 0.7670
2022-12-05 19:36:42 - Euclidean-Distance:       Pearson: 0.7872 Spearman: 0.7762
2022-12-05 19:36:42 - Dot-Product-Similarity:   Pearson: 0.7339 Spearman: 0.7125
2022-12-05 19:36:42 - Save model to models/reduced-256/msmarco-distilbert-base-tas-b
██████████████████████████████████████████████████████████████████████████████
2022-12-05 19:36:43 - Load pretrained SentenceTransformer: sentence-transformers/msmarco-distilbert-cos-v5
2022-12-05 19:36:44 - Use pytorch device: cuda
2022-12-05 19:36:44 - Read STSbenchmark test dataset
2022-12-05 19:36:45 - Original model performance:
2022-12-05 19:36:45 - EmbeddingSimilarityEvaluator: Evaluating the model on sts-benchmark-test dataset:
2022-12-05 19:36:50 - Cosine-Similarity :       Pearson: 0.7950 Spearman: 0.7824
2022-12-05 19:36:50 - Manhattan-Distance:       Pearson: 0.7854 Spearman: 0.7821
2022-12-05 19:36:50 - Euclidean-Distance:       Pearson: 0.7857 Spearman: 0.7824
2022-12-05 19:36:50 - Dot-Product-Similarity:   Pearson: 0.7950 Spearman: 0.7824
Batches: 100%|| 625/625 [00:37<00:00, 16.67it/s]
2022-12-05 19:37:36 - Model with 256 dimensions:
2022-12-05 19:37:36 - EmbeddingSimilarityEvaluator: Evaluating the model on sts-benchmark-test dataset:
2022-12-05 19:37:41 - Cosine-Similarity :       Pearson: 0.7946 Spearman: 0.7833
2022-12-05 19:37:41 - Manhattan-Distance:       Pearson: 0.7696 Spearman: 0.7685
2022-12-05 19:37:41 - Euclidean-Distance:       Pearson: 0.7898 Spearman: 0.7837
2022-12-05 19:37:41 - Dot-Product-Similarity:   Pearson: 0.7889 Spearman: 0.7770
2022-12-05 19:37:41 - Save model to models/reduced-256/sentence-transformers/msmarco-distilbert-cos-v5
██████████████████████████████████████████████████████████████████████████████
2022-12-05 19:37:42 - Load pretrained SentenceTransformer: sentence-transformers/msmarco-bert-base-dot-v5
2022-12-05 19:37:44 - Use pytorch device: cuda
2022-12-05 19:37:44 - Read STSbenchmark test dataset
2022-12-05 19:37:44 - Original model performance:
2022-12-05 19:37:44 - EmbeddingSimilarityEvaluator: Evaluating the model on sts-benchmark-test dataset:
2022-12-05 19:37:55 - Cosine-Similarity :       Pearson: 0.7682 Spearman: 0.7607
2022-12-05 19:37:55 - Manhattan-Distance:       Pearson: 0.7601 Spearman: 0.7539
2022-12-05 19:37:55 - Euclidean-Distance:       Pearson: 0.7608 Spearman: 0.7546
2022-12-05 19:37:55 - Dot-Product-Similarity:   Pearson: 0.6881 Spearman: 0.6961
Batches: 100%|| 625/625 [01:10<00:00,  8.92it/s]
2022-12-05 19:39:14 - Model with 256 dimensions:
2022-12-05 19:39:14 - EmbeddingSimilarityEvaluator: Evaluating the model on sts-benchmark-test dataset:
2022-12-05 19:39:25 - Cosine-Similarity :       Pearson: 0.7881 Spearman: 0.7760
2022-12-05 19:39:25 - Manhattan-Distance:       Pearson: 0.7438 Spearman: 0.7388
2022-12-05 19:39:25 - Euclidean-Distance:       Pearson: 0.7673 Spearman: 0.7595
2022-12-05 19:39:25 - Dot-Product-Similarity:   Pearson: 0.6638 Spearman: 0.6627
██████████████████████████████████████████████████████████████████████████████
2022-12-05 19:39:25 - Save model to models/reduced-256/sentence-transformers/msmarco-bert-base-dot-v5
2022-12-05 19:39:27 - Load pretrained SentenceTransformer: msmarco-distilbert-base-tas-b
2022-12-05 19:39:28 - Use pytorch device: cuda
2022-12-05 19:39:28 - Read STSbenchmark test dataset
2022-12-05 19:39:28 - Original model performance:
2022-12-05 19:39:28 - EmbeddingSimilarityEvaluator: Evaluating the model on sts-benchmark-test dataset:
2022-12-05 19:39:34 - Cosine-Similarity :       Pearson: 0.7913 Spearman: 0.7839
2022-12-05 19:39:34 - Manhattan-Distance:       Pearson: 0.7880 Spearman: 0.7789
2022-12-05 19:39:34 - Euclidean-Distance:       Pearson: 0.7886 Spearman: 0.7794
2022-12-05 19:39:34 - Dot-Product-Similarity:   Pearson: 0.7671 Spearman: 0.7501
Batches: 100%|| 625/625 [00:30<00:00, 20.46it/s]
2022-12-05 19:40:17 - Model with 512 dimensions:
2022-12-05 19:40:17 - EmbeddingSimilarityEvaluator: Evaluating the model on sts-benchmark-test dataset:
2022-12-05 19:40:19 - Cosine-Similarity :       Pearson: 0.7972 Spearman: 0.7877
2022-12-05 19:40:19 - Manhattan-Distance:       Pearson: 0.7630 Spearman: 0.7582
2022-12-05 19:40:19 - Euclidean-Distance:       Pearson: 0.7886 Spearman: 0.7792
2022-12-05 19:40:19 - Dot-Product-Similarity:   Pearson: 0.7607 Spearman: 0.7430
2022-12-05 19:40:19 - Save model to models/reduced-512/msmarco-distilbert-base-tas-b
██████████████████████████████████████████████████████████████████████████████
2022-12-05 19:40:20 - Load pretrained SentenceTransformer: sentence-transformers/msmarco-distilbert-cos-v5
2022-12-05 19:40:21 - Use pytorch device: cuda
2022-12-05 19:40:21 - Read STSbenchmark test dataset
2022-12-05 19:40:22 - Original model performance:
2022-12-05 19:40:22 - EmbeddingSimilarityEvaluator: Evaluating the model on sts-benchmark-test dataset:
2022-12-05 19:40:24 - Cosine-Similarity :       Pearson: 0.7950 Spearman: 0.7824
2022-12-05 19:40:24 - Manhattan-Distance:       Pearson: 0.7854 Spearman: 0.7821
2022-12-05 19:40:24 - Euclidean-Distance:       Pearson: 0.7857 Spearman: 0.7824
2022-12-05 19:40:24 - Dot-Product-Similarity:   Pearson: 0.7950 Spearman: 0.7824
Batches: 100%|| 625/625 [00:18<00:00, 33.66it/s]
2022-12-05 19:40:53 - Model with 512 dimensions:
2022-12-05 19:40:53 - EmbeddingSimilarityEvaluator: Evaluating the model on sts-benchmark-test dataset:
2022-12-05 19:40:56 - Cosine-Similarity :       Pearson: 0.7951 Spearman: 0.7827
2022-12-05 19:40:56 - Manhattan-Distance:       Pearson: 0.7478 Spearman: 0.7504
2022-12-05 19:40:56 - Euclidean-Distance:       Pearson: 0.7871 Spearman: 0.7830
2022-12-05 19:40:56 - Dot-Product-Similarity:   Pearson: 0.7955 Spearman: 0.7835
2022-12-05 19:40:56 - Save model to models/reduced-512/sentence-transformers/msmarco-distilbert-cos-v5
██████████████████████████████████████████████████████████████████████████████
2022-12-05 19:40:57 - Load pretrained SentenceTransformer: sentence-transformers/msmarco-bert-base-dot-v5
2022-12-05 19:41:00 - Use pytorch device: cuda
2022-12-05 19:41:00 - Read STSbenchmark test dataset
2022-12-05 19:41:00 - Original model performance:
2022-12-05 19:41:00 - EmbeddingSimilarityEvaluator: Evaluating the model on sts-benchmark-test dataset:
2022-12-05 19:41:05 - Cosine-Similarity :       Pearson: 0.7682 Spearman: 0.7607
2022-12-05 19:41:05 - Manhattan-Distance:       Pearson: 0.7601 Spearman: 0.7539
2022-12-05 19:41:05 - Euclidean-Distance:       Pearson: 0.7608 Spearman: 0.7546
2022-12-05 19:41:05 - Dot-Product-Similarity:   Pearson: 0.6881 Spearman: 0.6961
Batches: 100%|| 625/625 [00:33<00:00, 18.45it/s]
2022-12-05 19:41:51 - Model with 512 dimensions:
2022-12-05 19:41:51 - EmbeddingSimilarityEvaluator: Evaluating the model on sts-benchmark-test dataset:
2022-12-05 19:41:56 - Cosine-Similarity :       Pearson: 0.7768 Spearman: 0.7671
2022-12-05 19:41:56 - Manhattan-Distance:       Pearson: 0.7088 Spearman: 0.7075
2022-12-05 19:41:56 - Euclidean-Distance:       Pearson: 0.7615 Spearman: 0.7553
2022-12-05 19:41:56 - Dot-Product-Similarity:   Pearson: 0.6795 Spearman: 0.6900
2022-12-05 19:41:56 - Save model to models/reduced-512/sentence-transformers/msmarco-bert-base-dot-v5
██████████████████████████████████████████████████████████████████████████████