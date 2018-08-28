#!/bin/bash

# (train|add|add+train)-(cbow|skip)(100|300)-(nolda|ldafeat|ldafilt)

############
# SKIPGRAM #
############

# train-skip100-nolda
time ./emotiklue_keras.py train -m train-skip100-nolda -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_skipgram_100.txt.gz --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m train-skip100-nolda /ccl/projects/IEST/test_merged.csv
# train-skip100-ldafeat
time ./emotiklue_keras.py train -m train-skip100-ldafeat -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_skipgram_100.txt.gz --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --lda-mode feature --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m train-skip100-ldafeat --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv
# train-skip100-ldafilt
time ./emotiklue_keras.py train -m train-skip100-ldafilt -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_skipgram_100.txt.gz --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --lda-mode filter --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m train-skip100-ldafilt --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv

# train-skip300-nolda
time ./emotiklue_keras.py train -m train-skip300-nolda -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_skipgram_300.txt.gz --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m train-skip300-nolda /ccl/projects/IEST/test_merged.csv
# train-skip300-ldafeat
time ./emotiklue_keras.py train -m train-skip300-ldafeat -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_skipgram_300.txt.gz --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --lda-mode feature --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m train-skip300-ldafeat --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv
# train-skip300-ldafilt
time ./emotiklue_keras.py train -m train-skip300-ldafilt -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_skipgram_300.txt.gz --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --lda-mode filter --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m train-skip300-ldafilt --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv

# add-skip100-nolda
time ./emotiklue_keras.py train -m add-skip100-nolda -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_skipgram_100.txt.gz --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/additional_training_data_tweets+encow.csv
./emotiklue_keras.py test -m add-skip100-nolda /ccl/projects/IEST/test_merged.csv
# add-skip100-ldafeat
time ./emotiklue_keras.py train -m add-skip100-ldafeat -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_skipgram_100.txt.gz --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --lda-mode feature --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/additional_training_data_tweets+encow.csv
./emotiklue_keras.py test -m add-skip100-ldafeat --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv
# add-skip100-ldafilt
time ./emotiklue_keras.py train -m add-skip100-ldafilt -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_skipgram_100.txt.gz --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --lda-mode filter --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/additional_training_data_tweets+encow.csv
./emotiklue_keras.py test -m add-skip100-ldafilt --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv

# add-skip300-nolda
time ./emotiklue_keras.py train -m add-skip300-nolda -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_skipgram_300.txt.gz --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/additional_training_data_tweets+encow.csv
./emotiklue_keras.py test -m add-skip300-nolda /ccl/projects/IEST/test_merged.csv
# add-skip300-ldafeat
time ./emotiklue_keras.py train -m add-skip300-ldafeat -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_skipgram_300.txt.gz --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --lda-mode feature --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/additional_training_data_tweets+encow.csv
./emotiklue_keras.py test -m add-skip300-ldafeat --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv
# add-skip300-ldafilt
time ./emotiklue_keras.py train -m add-skip300-ldafilt -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_skipgram_300.txt.gz --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --lda-mode filter --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/additional_training_data_tweets+encow.csv
./emotiklue_keras.py test -m add-skip300-ldafilt --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv

# add+train-skip100-nolda
time ./emotiklue_keras.py retrain -m add-skip100-nolda -e 20 --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m add-skip100-nolda_retrain /ccl/projects/IEST/test_merged.csv
# add+train-skip100-ldafeat
time ./emotiklue_keras.py retrain -m add-skip100-ldafeat -e 20 --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m add-skip100-ldafeat_retrain --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv
# add+train-skip100-ldafilt
time ./emotiklue_keras.py retrain -m add-skip100-ldafilt -e 20 --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m add-skip100-ldafilt_retrain --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv

# add+train-skip300-nolda
time ./emotiklue_keras.py retrain -m add-skip300-nolda -e 20 --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m add-skip300-nolda_retrain /ccl/projects/IEST/test_merged.csv
# add+train-skip300-ldafeat
time ./emotiklue_keras.py retrain -m add-skip300-ldafeat -e 20 --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m add-skip300-ldafeat_retrain --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv
# add+train-skip300-ldafilt
time ./emotiklue_keras.py retrain -m add-skip300-ldafilt -e 20 --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m add-skip300-ldafilt_retrain --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv



########
# CBOW #
########

# train-cbow100-nolda
time ./emotiklue_keras.py train -m train-cbow100-nolda -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_cbow_100.txt.gz --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m train-cbow100-nolda /ccl/projects/IEST/test_merged.csv
# train-cbow100-ldafeat
time ./emotiklue_keras.py train -m train-cbow100-ldafeat -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_cbow_100.txt.gz --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --lda-mode feature --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m train-cbow100-ldafeat --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv
# train-cbow100-ldafilt
time ./emotiklue_keras.py train -m train-cbow100-ldafilt -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_cbow_100.txt.gz --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --lda-mode filter --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m train-cbow100-ldafilt --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv

# train-cbow300-nolda
time ./emotiklue_keras.py train -m train-cbow300-nolda -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_cbow_300.txt.gz --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m train-cbow300-nolda /ccl/projects/IEST/test_merged.csv
# train-cbow300-ldafeat
time ./emotiklue_keras.py train -m train-cbow300-ldafeat -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_cbow_300.txt.gz --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --lda-mode feature --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m train-cbow300-ldafeat --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv
# train-cbow300-ldafilt
time ./emotiklue_keras.py train -m train-cbow300-ldafilt -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_cbow_300.txt.gz --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --lda-mode filter --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m train-cbow300-ldafilt --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv

# add-cbow100-nolda
time ./emotiklue_keras.py train -m add-cbow100-nolda -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_cbow_100.txt.gz --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/additional_training_data_tweets+encow.csv
./emotiklue_keras.py test -m add-cbow100-nolda /ccl/projects/IEST/test_merged.csv
# add-cbow100-ldafeat
time ./emotiklue_keras.py train -m add-cbow100-ldafeat -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_cbow_100.txt.gz --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --lda-mode feature --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/additional_training_data_tweets+encow.csv
./emotiklue_keras.py test -m add-cbow100-ldafeat --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv
# add-cbow100-ldafilt
time ./emotiklue_keras.py train -m add-cbow100-ldafilt -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_cbow_100.txt.gz --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --lda-mode filter --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/additional_training_data_tweets+encow.csv
./emotiklue_keras.py test -m add-cbow100-ldafilt --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv

# add-cbow300-nolda
time ./emotiklue_keras.py train -m add-cbow300-nolda -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_cbow_300.txt.gz --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/additional_training_data_tweets+encow.csv
./emotiklue_keras.py test -m add-cbow300-nolda /ccl/projects/IEST/test_merged.csv
# add-cbow300-ldafeat
time ./emotiklue_keras.py train -m add-cbow300-ldafeat -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_cbow_300.txt.gz --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --lda-mode feature --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/additional_training_data_tweets+encow.csv
./emotiklue_keras.py test -m add-cbow300-ldafeat --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv
# add-cbow300-ldafilt
time ./emotiklue_keras.py train -m add-cbow300-ldafilt -e 20 --embeddings /ccl/projects/IEST/tweets_dedup_w2v_cbow_300.txt.gz --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --lda-mode filter --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/additional_training_data_tweets+encow.csv
./emotiklue_keras.py test -m add-cbow300-ldafilt --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv

# add+train-cbow100-nolda
time ./emotiklue_keras.py retrain -m add-cbow100-nolda -e 20 --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m add-cbow100-nolda_retrain /ccl/projects/IEST/test_merged.csv
# add+train-cbow100-ldafeat
time ./emotiklue_keras.py retrain -m add-cbow100-ldafeat -e 20 --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m add-cbow100-ldafeat_retrain --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv
# add+train-cbow100-ldafilt
time ./emotiklue_keras.py retrain -m add-cbow100-ldafilt -e 20 --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m add-cbow100-ldafilt_retrain --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv

# add+train-cbow300-nolda
time ./emotiklue_keras.py retrain -m add-cbow300-nolda -e 20 --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m add-cbow300-nolda_retrain /ccl/projects/IEST/test_merged.csv
# add+train-cbow300-ldafeat
time ./emotiklue_keras.py retrain -m add-cbow300-ldafeat -e 20 --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m add-cbow300-ldafeat_retrain --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv
# add+train-cbow300-ldafilt
time ./emotiklue_keras.py retrain -m add-cbow300-ldafilt -e 20 --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt --val /ccl/projects/IEST/dev-v3.csv /ccl/projects/IEST/train-v3.csv_tokenized.txt
./emotiklue_keras.py test -m add-cbow300-ldafilt_retrain --lda /ccl/projects/IEST/lda/tweets10m_lda_100 --dict /ccl/projects/IEST/lda/tweets10m_dict.txt /ccl/projects/IEST/test_merged.csv
