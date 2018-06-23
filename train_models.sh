#!/bin/bash

time ./emotiklue_keras.py train -m model_train_w2v_skipgram --val /ccl/projects/IEST/dev-v3.csv --embeddings /ccl/projects/IEST/tweets_w2v_skipgram_100.txt.gz /ccl/projects/IEST/train-v3.csv_tokenized.txt
time ./emotiklue_keras.py train -m model_train_w2v_cbow --val /ccl/projects/IEST/dev-v3.csv --embeddings /ccl/projects/IEST/tweets_w2v_cbow_100.txt.gz /ccl/projects/IEST/train-v3.csv_tokenized.txt
time ./emotiklue_keras.py train -m model_train_glove --val /ccl/projects/IEST/dev-v3.csv --embeddings /cip/corpora/DSM/GloVe/glove.twitter.27B.100d.txt.gz /ccl/projects/IEST/train-v3.csv_tokenized.txt

time ./emotiklue_keras.py train -m model_extra_w2v_skipgram --val /ccl/projects/IEST/dev-v3.csv --embeddings /ccl/projects/IEST/tweets_w2v_skipgram_100.txt.gz /ccl/projects/IEST/additional_training_data_balanced.csv
time ./emotiklue_keras.py train -m model_extra_w2v_cbow --val /ccl/projects/IEST/dev-v3.csv --embeddings /ccl/projects/IEST/tweets_w2v_cbow_100.txt.gz /ccl/projects/IEST/additional_training_data_balanced.csv
time ./emotiklue_keras.py train -m model_extra_glove --val /ccl/projects/IEST/dev-v3.csv --embeddings /cip/corpora/DSM/GloVe/glove.twitter.27B.100d.txt.gz /ccl/projects/IEST/additional_training_data_balanced.csv

time ./emotiklue_keras.py retrain -m model_extra_w2v_skipgram --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
time ./emotiklue_keras.py retrain -m model_extra_w2v_cbow --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
time ./emotiklue_keras.py retrain -m model_extra_glove --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
