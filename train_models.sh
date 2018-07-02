#!/bin/bash

time ./emotiklue_keras.py train -m model_train_w2v_t+e_skipgram --val /ccl/projects/IEST/dev-v3.csv --embeddings /ccl/projects/IEST/tweets+encow_w2v_skipgram_100.txt.gz /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m model_train_w2v_t+e_skipgram /ccl/projects/IEST/dev-v3.csv
time ./emotiklue_keras.py train -m model_train_w2v_t+e_cbow --val /ccl/projects/IEST/dev-v3.csv --embeddings /ccl/projects/IEST/tweets+encow_w2v_cbow_100.txt.gz /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m model_train_w2v_t+e_cbow /ccl/projects/IEST/dev-v3.csv
# time ./emotiklue_keras.py train -m model_train_w2v_skipgram --val /ccl/projects/IEST/dev-v3.csv --embeddings /ccl/projects/IEST/tweets_w2v_skipgram_100.txt.gz /ccl/projects/IEST/train-v3.csv_tokenized.txt
# ./emotiklue_keras.py test -m model_train_w2v_skipgram /ccl/projects/IEST/dev-v3.csv
# time ./emotiklue_keras.py train -m model_train_w2v_cbow --val /ccl/projects/IEST/dev-v3.csv --embeddings /ccl/projects/IEST/tweets_w2v_cbow_100.txt.gz /ccl/projects/IEST/train-v3.csv_tokenized.txt
# ./emotiklue_keras.py test -m model_train_w2v_cbow /ccl/projects/IEST/dev-v3.csv
# time ./emotiklue_keras.py train -m model_train_glove --val /ccl/projects/IEST/dev-v3.csv --embeddings /cip/corpora/DSM/GloVe/glove.twitter.27B.100d.txt.gz /ccl/projects/IEST/train-v3.csv_tokenized.txt
# ./emotiklue_keras.py test -m model_train_glove /ccl/projects/IEST/dev-v3.csv

time ./emotiklue_keras.py train -m model_extra_w2v_t+e_skipgram --val /ccl/projects/IEST/dev-v3.csv --embeddings /ccl/projects/IEST/tweets+encow_w2v_skipgram_100.txt.gz /ccl/projects/IEST/additional_training_data_tweets+encow.csv
./emotiklue_keras.py test -m model_extra_w2v_t+e_skipgram /ccl/projects/IEST/dev-v3.csv
time ./emotiklue_keras.py train -m model_extra_w2v_t+e_cbow --val /ccl/projects/IEST/dev-v3.csv --embeddings /ccl/projects/IEST/tweets_w2v_cbow+encow_100.txt.gz /ccl/projects/IEST/additional_training_data_tweets+encow.csv
./emotiklue_keras.py test -m model_extra_w2v_t+e_cbow /ccl/projects/IEST/dev-v3.csv
# time ./emotiklue_keras.py train -m model_extra_w2v_skipgram --val /ccl/projects/IEST/dev-v3.csv --embeddings /ccl/projects/IEST/tweets_w2v_skipgram_100.txt.gz /ccl/projects/IEST/additional_training_data_tweets+encow.csv
# ./emotiklue_keras.py test -m model_extra_w2v_skipgram /ccl/projects/IEST/dev-v3.csv
# time ./emotiklue_keras.py train -m model_extra_w2v_cbow --val /ccl/projects/IEST/dev-v3.csv --embeddings /ccl/projects/IEST/tweets_w2v_cbow_100.txt.gz /ccl/projects/IEST/additional_training_data_tweets+encow.csv
# ./emotiklue_keras.py test -m model_extra_w2v_cbow /ccl/projects/IEST/dev-v3.csv
# time ./emotiklue_keras.py train -m model_extra_glove --val /ccl/projects/IEST/dev-v3.csv --embeddings /cip/corpora/DSM/GloVe/glove.twitter.27B.100d.txt.gz /ccl/projects/IEST/additional_training_data_tweets+encow.csv
# ./emotiklue_keras.py test -m model_extra_glove /ccl/projects/IEST/dev-v3.csv

time ./emotiklue_keras.py adapt -m model_extra_w2v_t+e_skipgram --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m model_extra_w2v_t+e_skipgram_adapt /ccl/projects/IEST/dev-v3.csv
time ./emotiklue_keras.py adapt -m model_extra_w2v_t+e_cbow --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m model_extra_w2v_t+e_cbow_adapt /ccl/projects/IEST/dev-v3.csv
# time ./emotiklue_keras.py adapt -m model_extra_w2v_skipgram --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
# ./emotiklue_keras.py test -m model_extra_w2v_skipgram_adapt /ccl/projects/IEST/dev-v3.csv
# time ./emotiklue_keras.py adapt -m model_extra_w2v_cbow --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
# ./emotiklue_keras.py test -m model_extra_w2v_cbow_adapt /ccl/projects/IEST/dev-v3.csv
# time ./emotiklue_keras.py adapt -m model_extra_glove --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
# ./emotiklue_keras.py test -m model_extra_glove_adapt /ccl/projects/IEST/dev-v3.csv

time ./emotiklue_keras.py retrain -m model_extra_w2v_t+e_skipgram --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m model_extra_w2v_t+e_skipgram_retrain /ccl/projects/IEST/dev-v3.csv
time ./emotiklue_keras.py retrain -m model_extra_w2v_t+e_cbow --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m model_extra_w2v_t+e_cbow_retrain /ccl/projects/IEST/dev-v3.csv
# time ./emotiklue_keras.py retrain -m model_extra_w2v_skipgram --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
# ./emotiklue_keras.py test -m model_extra_w2v_skipgram_retrain /ccl/projects/IEST/dev-v3.csv
# time ./emotiklue_keras.py retrain -m model_extra_w2v_cbow --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
# ./emotiklue_keras.py test -m model_extra_w2v_cbow_retrain /ccl/projects/IEST/dev-v3.csv
# time ./emotiklue_keras.py retrain -m model_extra_glove --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
# ./emotiklue_keras.py test -m model_extra_glove_retrain /ccl/projects/IEST/dev-v3.csv
