#!/usr/bin/env python3

import argparse
import gzip
import itertools
import json
import os
import unicodedata

import gensim
import keras.backend as K
import keras.layers
import keras.models
import keras.preprocessing.sequence
import keras.regularizers
import keras.utils
import keras.utils.generic_utils
import numpy as np
import tensorflow as tf

import evaluate_iest


class L1L2_m(keras.regularizers.Regularizer):
    """Regularizer for L1 and L2 regularization.
    # Arguments
    l1: Float; L1 regularization factor.
    l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0.0, l2=0.01, prior=None, prior_shape=None):
        with K.name_scope(self.__class__.__name__):
            self.l1 = K.variable(l1, name='l1')
            self.l2 = K.variable(l2, name='l2')
            self.val_prior = prior
            if prior is None:
                prior = np.zeros(prior_shape)
                self.prior_shape = prior_shape
            if type(prior) == dict:
                prior_array = np.array(prior["value"])
                self.prior = K.variable(prior_array, name="prior")
                self.prior_shape = prior_array.shape
            else:
                self.prior = K.variable(prior, name="prior")
                self.prior_shape = prior.shape
            self.val_l1 = l1
            self.val_l2 = l2

    def set_l1_l2(self, l1, l2):
        K.set_value(self.l1, l1)
        K.set_value(self.l2, l2)
        self.val_l1 = l1
        self.val_l2 = l2

    def set_prior(self, prior):
        K.set_value(self.prior, prior)
        self.val_prior = prior
        self.prior_shape = prior.shape

    def __call__(self, x):
        regularization = 0.
        if self.val_l1 > 0.:
            if self.val_prior is not None:
                regularization += K.sum(self.l1 * K.abs(x - self.prior))
            else:
                regularization += K.sum(self.l1 * K.abs(x))
        if self.val_l2 > 0.:
            if self.val_prior is not None:
                regularization += K.sum(self.l2 * K.square(x - self.prior))
            else:
                regularization += K.sum(self.l2 * K.square(x))
        return regularization

    def get_config(self):
        config = {'l1': float(K.get_value(self.l1)),
                  'l2': float(K.get_value(self.l2)),
                  'prior_shape': self.prior_shape}
        if self.val_prior is not None:
            config["prior"] = K.get_value(self.prior)
        return config


def arguments():
    parser = argparse.ArgumentParser("EmotiKLUE")
    subparsers = parser.add_subparsers()
    parser_train = subparsers.add_parser("train", help="train a model (run 'train -h' for more details)")
    # parser_adapt = subparsers.add_parser("adapt", help="adapt a model (run 'retrain -h' for more details)")
    parser_retrain = subparsers.add_parser("retrain", help="retrain a model (run 'retrain -h' for more details)")
    parser_test = subparsers.add_parser("test", help="test a model (run 'test -h' for more details)")
    parser_train.add_argument("-m", "--model", type=os.path.abspath, required=True, help="Path to model")
    parser_train.add_argument("--val", type=os.path.abspath, required=True, help="Dataset for validation")
    parser_train.add_argument("--embeddings", type=os.path.abspath, required=True, help="Word embeddings")
    parser_train.add_argument("--chars", action="store_true", help="Use additional character-level LSTMs")
    parser_train.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser_train.add_argument("--lda", type=str, required=True, help="LDA model")
    parser_train.add_argument("--dict", type=os.path.abspath, required=True, help="Dictionary for LDA")
    # parser_train.add_argument("--gpu", type=int, default=0, help="Number of GPUs to use")
    parser_train.add_argument("FILE", type=os.path.abspath, help="Dataset for training")
    parser_train.set_defaults(func=train)
    parser_retrain.add_argument("-m", "--model", type=os.path.abspath, required=True, help="Path to model")
    parser_retrain.add_argument("--val", type=os.path.abspath, required=True, help="Dataset for validation")
    # parser_retrain.add_argument("--gpu", type=int, default=0, help="Number of GPUs to use")
    parser_retrain.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser_retrain.add_argument("--lda", type=str, required=True, help="LDA model")
    parser_retrain.add_argument("--dict", type=os.path.abspath, required=True, help="Dictionary for LDA")
    parser_retrain.add_argument("FILE", type=os.path.abspath, help="Dataset for retraining")
    parser_retrain.set_defaults(func=retrain)
    # parser_adapt.add_argument("-m", "--model", type=os.path.abspath, required=True, help="Path to model")
    # parser_adapt.add_argument("--val", type=os.path.abspath, required=True, help="Dataset for validation")
    # # parser_adapt.add_argument("--gpu", type=int, default=0, help="Number of GPUs to use")
    # parser_adapt.add_argument("FILE", type=os.path.abspath, help="Dataset for adapting")
    # parser_adapt.set_defaults(func=regularized_adaptation)
    parser_test.add_argument("-m", "--model", type=os.path.abspath, required=True, help="Path to model")
    parser_test.add_argument("--lda", type=str, required=True, help="LDA model")
    parser_test.add_argument("--dict", type=os.path.abspath, required=True, help="Dictionary for LDA")
    parser_test.add_argument("FILE", type=os.path.abspath, help="Dataset for testing")
    parser_test.set_defaults(func=test)
    return parser.parse_args()


def read_dataset(filename):
    data = []
    vocabulary, classes = set(), set()
    with open(filename, encoding="utf8") as fh:
        for line in fh:
            if "[#TRIGGERWORD#]" not in line:
                continue
            cls, text = line.strip().split("\t")
            left_str, right_str = text.strip().split("[#TRIGGERWORD#]")
            left_words = left_str.strip().split()
            right_words = right_str.strip().split()
            left_str = unicodedata.normalize("NFD", left_str)
            right_str = unicodedata.normalize("NFD", right_str)
            vocabulary.update(set(itertools.chain(left_words, right_words)))
            classes.add(cls)
            data.append((left_words, right_words, left_str, right_str, cls))
    lw, rw, lc, rc, tgt = zip(*data)
    return lw, rw, lc, rc, tgt, vocabulary, classes


def read_glove(filename):
    embeddings_index = {}
    with gzip.open(filename, mode="rt", encoding="utf8") as fh:
        for line in fh:
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


def vectorize_characters(sequence, reverse=False):
    if reverse:
        return list(reversed(sequence.encode()))
    else:
        return list(sequence.encode())


def vectorize_words(sequence, mapping, reverse=False):
    if reverse:
        sequence = reversed(sequence)
    return [mapping.get(w, len(mapping) + 1) for w in sequence]


def topic_distribution(dict_path, lda_prefix, left_words, right_words):
    dictionary = gensim.corpora.Dictionary.load_from_text(dict_path)
    lda = gensim.models.ldamodel.LdaModel.load(lda_prefix)
    topic_dist = []
    for lw, rw in zip(left_words, right_words):
        bow = dictionary.doc2bow(lw + rw)
        topics = lda[bow]
        td = [0.00001] * 100
        for i, p in topics:
            td[i] = p
        topic_dist.append(td)
    return np.array(topic_dist)


def train(args):
    WORD_EMBEDDING_DIM = 300
    CHAR_EMBEDDING_DIM = 16
    WORD_LSTM_DIM = 100
    CHAR_LSTM_DIM = CHAR_EMBEDDING_DIM
    if args.chars:
        DENSE_DIM = WORD_LSTM_DIM + CHAR_LSTM_DIM
    else:
        DENSE_DIM = WORD_LSTM_DIM
    DROPOUT = 0.2
    RECURRENT_DROPOUT = 0.0
    BATCH_SIZE = 160

    train_lw, train_rw, train_lc, train_rc, train_tgt, vocabulary, classes = read_dataset(args.FILE)
    val_lw, val_rw, val_lc, val_rc, val_tgt, _, _ = read_dataset(args.val)
    embeddings_index = read_glove(args.embeddings)

    # mappings
    word_to_idx = {w: i for i, w in enumerate(sorted(vocabulary), start=1)}
    tgt_to_idx = {c: i for i, c in enumerate(sorted(classes), start=0)}

    # create embedding layers
    embedding_matrix = np.zeros((len(vocabulary) + 2, WORD_EMBEDDING_DIM))
    for word, idx in word_to_idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[idx] = embedding_vector

    # LDA models
    train_topics = topic_distribution(args.dict, args.lda, train_lw, train_rw)
    val_topics = topic_distribution(args.dict, args.lda, val_lw, val_rw)

    # vectorize
    train_lw = [vectorize_words(lw, word_to_idx) for lw in train_lw]
    train_rw = [vectorize_words(rw, word_to_idx, reverse=True) for rw in train_rw]
    train_lc = [vectorize_characters(lc) for lc in train_lc]
    train_rc = [vectorize_characters(rc, reverse=True) for rc in train_rc]
    train_tgt = vectorize_words(train_tgt, tgt_to_idx)
    targets = keras.utils.to_categorical(train_tgt, num_classes=len(classes))
    val_lw = [vectorize_words(lw, word_to_idx) for lw in val_lw]
    val_rw = [vectorize_words(rw, word_to_idx, reverse=True) for rw in val_rw]
    val_lc = [vectorize_characters(lc) for lc in val_lc]
    val_rc = [vectorize_characters(rc, reverse=True) for rc in val_rc]
    val_tgt = vectorize_words(val_tgt, tgt_to_idx)
    val_targets = keras.utils.to_categorical(val_tgt, num_classes=len(classes))

    # pad sequences
    max_len_lw = int(max((len(lw) for lw in itertools.chain(train_lw, val_lw))) * 1.1)
    max_len_rw = int(max((len(rw) for rw in itertools.chain(train_rw, val_rw))) * 1.1)
    max_len_lc = int(max((len(lc) for lc in itertools.chain(train_lc, val_lc))) * 1.1)
    max_len_rc = int(max((len(rc) for rc in itertools.chain(train_rc, val_rc))) * 1.1)

    train_left_words = keras.preprocessing.sequence.pad_sequences(train_lw, maxlen=max_len_lw, padding="pre", truncating="pre")
    train_right_words = keras.preprocessing.sequence.pad_sequences(train_rw, maxlen=max_len_rw, padding="pre", truncating="pre")
    train_left_chars = keras.preprocessing.sequence.pad_sequences(train_lc, maxlen=max_len_lc, padding="pre", truncating="pre")
    train_right_chars = keras.preprocessing.sequence.pad_sequences(train_rc, maxlen=max_len_rc, padding="pre", truncating="pre")
    val_left_words = keras.preprocessing.sequence.pad_sequences(val_lw, maxlen=max_len_lw, padding="pre", truncating="pre")
    val_right_words = keras.preprocessing.sequence.pad_sequences(val_rw, maxlen=max_len_rw, padding="pre", truncating="pre")
    val_left_chars = keras.preprocessing.sequence.pad_sequences(val_lc, maxlen=max_len_lc, padding="pre", truncating="pre")
    val_right_chars = keras.preprocessing.sequence.pad_sequences(val_rc, maxlen=max_len_rc, padding="pre", truncating="pre")

    # with keras.utils.CustomObjectScope({L1L2_m.__name__: L1L2_m}):
    # input layers
    input_lw = keras.layers.Input(shape=(train_left_words.shape[1],))
    input_rw = keras.layers.Input(shape=(train_right_words.shape[1],))
    input_lc = keras.layers.Input(shape=(train_left_chars.shape[1],))
    input_rc = keras.layers.Input(shape=(train_right_chars.shape[1],))
    input_topics = keras.layers.Input(shape=(100,))

    # embedding layers
    embedding_lw = keras.layers.Embedding(len(vocabulary) + 2, WORD_EMBEDDING_DIM, mask_zero=True, weights=[embedding_matrix], trainable=False)(input_lw)
    embedding_rw = keras.layers.Embedding(len(vocabulary) + 2, WORD_EMBEDDING_DIM, mask_zero=True, weights=[embedding_matrix], trainable=False)(input_rw)
    embedding_lc = keras.layers.Embedding(256, CHAR_EMBEDDING_DIM, mask_zero=True)(input_lc)
    embedding_rc = keras.layers.Embedding(256, CHAR_EMBEDDING_DIM, mask_zero=True)(input_rc)

    # LSTMs
    # lstm_lw = keras.layers.LSTM(WORD_LSTM_DIM, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT, kernel_regularizer=L1L2_m(l1=0, l2=0, prior_shape=(WORD_EMBEDDING_DIM, WORD_LSTM_DIM * 4)))(embedding_lw)
    # lstm_rw = keras.layers.LSTM(WORD_LSTM_DIM, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT, kernel_regularizer=L1L2_m(l1=0, l2=0, prior_shape=(WORD_EMBEDDING_DIM, WORD_LSTM_DIM * 4)))(embedding_rw)
    # lstm_lc = keras.layers.LSTM(CHAR_LSTM_DIM, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT, kernel_regularizer=L1L2_m(l1=0, l2=0, prior_shape=(CHAR_EMBEDDING_DIM, CHAR_LSTM_DIM * 4)))(embedding_lc)
    # lstm_rc = keras.layers.LSTM(CHAR_LSTM_DIM, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT, kernel_regularizer=L1L2_m(l1=0, l2=0, prior_shape=(CHAR_EMBEDDING_DIM, CHAR_LSTM_DIM * 4)))(embedding_rc)
    lstm_lw = keras.layers.LSTM(WORD_LSTM_DIM, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT)(embedding_lw)
    lstm_rw = keras.layers.LSTM(WORD_LSTM_DIM, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT)(embedding_rw)
    lstm_lc = keras.layers.LSTM(CHAR_LSTM_DIM, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT)(embedding_lc)
    lstm_rc = keras.layers.LSTM(CHAR_LSTM_DIM, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT)(embedding_rc)

    # concatenate
    if args.chars:
        lstm_out = keras.layers.Concatenate(axis=1)([lstm_lw, lstm_rw, lstm_lc, lstm_rc])
        LSTM_DIM = 2 * WORD_LSTM_DIM + 2 * CHAR_LSTM_DIM
    else:
        lstm_out = keras.layers.Concatenate(axis=1)([lstm_lw, lstm_rw])
        LSTM_DIM = 2 * WORD_LSTM_DIM

    # dense layers
    # dense01 = keras.layers.Dense(DENSE_DIM, activation="tanh", kernel_regularizer=L1L2_m(l1=0, l2=0, prior_shape=(LSTM_DIM, DENSE_DIM)))(lstm_out)

    ###########
    # NOTHING #
    ###########
    # dense01 = keras.layers.Dense(DENSE_DIM, activation="tanh")(lstm_out)
    # dropout01 = keras.layers.Dropout(DROPOUT)(dense01)
    # predictions = keras.layers.Dense(len(classes), activation="softmax")(dropout01)
    ###########
    # NOTHING #
    ###########

    ########################
    # MULTIPLY WITH TOPICS #
    ########################
    # # element-wise multiplication with topics
    # topic_filter = keras.layers.Multiply()([dropout01, input_topics])

    # # dense layer
    # dense02 = keras.layers.Dense(DENSE_DIM // 2, activation="tanh")(topic_filter)
    # dropout02 = keras.layers.Dropout(DROPOUT)(dense02)

    # predictions = keras.layers.Dense(len(classes), activation="softmax")(dropout02)
    ########################
    # MULTIPLY WITH TOPICS #
    ########################

    ######################
    # TOPICS AS FEATURES #
    ######################
    topics = keras.layers.Concatenate(axis=1)([lstm_out, input_topics])
    dense01 = keras.layers.Dense(DENSE_DIM, activation="tanh")(topics)
    dropout01 = keras.layers.Dropout(DROPOUT)(dense01)
    predictions = keras.layers.Dense(len(classes), activation="softmax")(dropout01)
    ######################
    # TOPICS AS FEATURES #
    ######################

    # dense02 = keras.layers.Dense(DENSE_DIM // 2, activation="tanh", kernel_regularizer=L1L2_m(l1=0, l2=0, prior_shape=(LSTM_DIM, DENSE_DIM)))(dropout01)
    # predictions = keras.layers.Dense(len(classes), activation="softmax", kernel_regularizer=L1L2_m(l1=0, l2=0, prior_shape=(DENSE_DIM, len(classes))))(dropout02)

    # predictions = keras.layers.Dense(len(classes), activation="softmax")(dropout01)

    with tf.device('/cpu:0'):
        if args.chars:
            model = keras.models.Model(inputs=[input_lw, input_rw, input_lc, input_rc, input_topics], outputs=predictions)
        else:
            model = keras.models.Model(inputs=[input_lw, input_rw, input_topics], outputs=predictions)
    # if args.gpu > 1:
    #     parallel_model = keras.utils.multi_gpu_model(model, gpus=args.gpu)
    #     parallel_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    #     parallel_model.fit([train_left_words, train_right_words, train_left_chars, train_right_chars], targets, batch_size=BATCH_SIZE * args.gpu, epochs=EPOCHS,
    #                        validation_data=([val_left_words, val_right_words, val_left_chars, val_right_chars], val_targets))
    # else:
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    if args.chars:
        model.fit([train_left_words, train_right_words, train_left_chars, train_right_chars, train_topics], targets, batch_size=BATCH_SIZE, epochs=args.epochs,
                  validation_data=([val_left_words, val_right_words, val_left_chars, val_right_chars, val_topics], val_targets))
    else:
        model.fit([train_left_words, train_right_words, train_topics], targets, batch_size=BATCH_SIZE, epochs=args.epochs,
                  validation_data=([val_left_words, val_right_words, val_topics], val_targets))
    model.save("%s.h5" % args.model)
    with open("%s.maps" % args.model, mode="w", encoding="utf-8") as f:
        json.dump((word_to_idx, tgt_to_idx, max_len_lw, max_len_rw, max_len_lc, max_len_rc, args.chars), f, ensure_ascii=False)

    # Official score
    if args.chars:
        predictions = model.predict([val_left_words, val_right_words, val_left_chars, val_right_chars, val_topics])
    else:
        predictions = model.predict([val_left_words, val_right_words, val_topics])
    idx_to_tgt = {i: c for c, i in tgt_to_idx.items()}
    predicted = [idx_to_tgt[p] for p in predictions.argmax(axis=1)]
    evaluate_iest.calculatePRF(list(val_targets), predicted)


def retrain(args):
    BATCH_SIZE = 160
    # with keras.utils.CustomObjectScope({L1L2_m.__name__: L1L2_m}):
    model = keras.models.load_model("%s.h5" % args.model)
    # model.optimizer.set_state()
    with open("%s.maps" % args.model, encoding="utf-8") as f:
        word_to_idx, tgt_to_idx, max_len_lw, max_len_rw, max_len_lc, max_len_rc, chars = json.load(f)
    train_lw, train_rw, train_lc, train_rc, train_tgt, _, _ = read_dataset(args.FILE)
    train_topics = topic_distribution(args.dict, args.lda, train_lw, train_rw)
    train_lw = [vectorize_words(lw, word_to_idx) for lw in train_lw]
    train_rw = [vectorize_words(rw, word_to_idx, reverse=True) for rw in train_rw]
    train_lc = [vectorize_characters(lc) for lc in train_lc]
    train_rc = [vectorize_characters(rc, reverse=True) for rc in train_rc]
    train_tgt = vectorize_words(train_tgt, tgt_to_idx)
    targets = keras.utils.to_categorical(train_tgt, num_classes=len(tgt_to_idx.values()))
    train_left_words = keras.preprocessing.sequence.pad_sequences(train_lw, maxlen=max_len_lw, padding="pre", truncating="pre")
    train_right_words = keras.preprocessing.sequence.pad_sequences(train_rw, maxlen=max_len_rw, padding="pre", truncating="pre")
    train_left_chars = keras.preprocessing.sequence.pad_sequences(train_lc, maxlen=max_len_lc, padding="pre", truncating="pre")
    train_right_chars = keras.preprocessing.sequence.pad_sequences(train_rc, maxlen=max_len_rc, padding="pre", truncating="pre")
    val_lw, val_rw, val_lc, val_rc, val_tgt, _, _ = read_dataset(args.val)
    val_topics = topic_distribution(args.dict, args.lda, val_lw, val_rw)
    val_lw = [vectorize_words(lw, word_to_idx) for lw in val_lw]
    val_rw = [vectorize_words(rw, word_to_idx, reverse=True) for rw in val_rw]
    val_lc = [vectorize_characters(lc) for lc in val_lc]
    val_rc = [vectorize_characters(rc, reverse=True) for rc in val_rc]
    val_tgt = vectorize_words(val_tgt, tgt_to_idx)
    val_targets = keras.utils.to_categorical(val_tgt, num_classes=len(tgt_to_idx.values()))
    val_left_words = keras.preprocessing.sequence.pad_sequences(val_lw, maxlen=max_len_lw, padding="pre", truncating="pre")
    val_right_words = keras.preprocessing.sequence.pad_sequences(val_rw, maxlen=max_len_rw, padding="pre", truncating="pre")
    val_left_chars = keras.preprocessing.sequence.pad_sequences(val_lc, maxlen=max_len_lc, padding="pre", truncating="pre")
    val_right_chars = keras.preprocessing.sequence.pad_sequences(val_rc, maxlen=max_len_rc, padding="pre", truncating="pre")
    if chars:
        model.fit([train_left_words, train_right_words, train_left_chars, train_right_chars, train_topics], targets, batch_size=BATCH_SIZE, epochs=args.epochs,
                  validation_data=([val_left_words, val_right_words, val_left_chars, val_right_chars, val_topics], val_targets))
    else:
        model.fit([train_left_words, train_right_words, train_topics], targets, batch_size=BATCH_SIZE, epochs=args.epochs,
                  validation_data=([val_left_words, val_right_words, val_topics], val_targets))
    model.save("%s_retrain.h5" % args.model)
    with open("%s_retrain.maps" % args.model, mode="w", encoding="utf-8") as f:
        json.dump((word_to_idx, tgt_to_idx, max_len_lw, max_len_rw, max_len_lc, max_len_rc, chars), f, ensure_ascii=False)

    # Official score
    if chars:
        predictions = model.predict([val_left_words, val_right_words, val_left_chars, val_right_chars, val_topics])
    else:
        predictions = model.predict([val_left_words, val_right_words, val_topics])
    idx_to_tgt = {i: c for c, i in tgt_to_idx.items()}
    predicted = [idx_to_tgt[p] for p in predictions.argmax(axis=1)]
    evaluate_iest.calculatePRF(list(val_targets), predicted)


def test(args):
    # with keras.utils.CustomObjectScope({L1L2_m.__name__: L1L2_m}):
    model = keras.models.load_model("%s.h5" % args.model)
    with open("%s.maps" % args.model, encoding="utf-8") as f:
        word_to_idx, tgt_to_idx, max_len_lw, max_len_rw, max_len_lc, max_len_rc, chars = json.load(f)
    idx_to_tgt = {i: c for c, i in tgt_to_idx.items()}
    test_lw, test_rw, test_lc, test_rc, test_tgt, _, _ = read_dataset(args.FILE)
    left_words = test_lw
    topics = topic_distribution(args.dict, args.lda, test_lw, test_rw)
    test_lw = [vectorize_words(lw, word_to_idx) for lw in test_lw]
    test_rw = [vectorize_words(rw, word_to_idx, reverse=True) for rw in test_rw]
    test_lc = [vectorize_characters(lc) for lc in test_lc]
    test_rc = [vectorize_characters(rc, reverse=True) for rc in test_rc]
    test_left_words = keras.preprocessing.sequence.pad_sequences(test_lw, maxlen=max_len_lw, padding="pre", truncating="pre")
    test_right_words = keras.preprocessing.sequence.pad_sequences(test_rw, maxlen=max_len_rw, padding="pre", truncating="pre")
    test_left_chars = keras.preprocessing.sequence.pad_sequences(test_lc, maxlen=max_len_lc, padding="pre", truncating="pre")
    test_right_chars = keras.preprocessing.sequence.pad_sequences(test_rc, maxlen=max_len_rc, padding="pre", truncating="pre")
    if chars:
        predictions = model.predict([test_left_words, test_right_words, test_left_chars, test_right_chars, topics])
    else:
        predictions = model.predict([test_left_words, test_right_words, topics])
    predicted = [idx_to_tgt[p] for p in predictions.argmax(axis=1)]
    evaluate_iest.calculatePRF(list(test_tgt), predicted)
    for i, lw in enumerate(left_words):
        if len(lw) > 0 and lw[-1] == "un":
            predicted[i] = "joy"
    evaluate_iest.calculatePRF(list(test_tgt), predicted)


def regularizer_factory(l2, prior, adapt=False):
    prior_weights = K.constant(prior)

    def null_regularizer(weight_matrix):
        return 0 * weight_matrix

    def regularizer(weight_matrix):
        return K.sum(l2 * K.square(weight_matrix - prior_weights))
    if adapt:
        return regularizer
    else:
        return null_regularizer


def nr(weight_matrix):
        return 0 * weight_matrix


def regularized_adaptation(args):
    BATCH_SIZE = 160
    with keras.utils.CustomObjectScope({L1L2_m.__name__: L1L2_m}):
        model = keras.models.load_model("%s.h5" % args.model)
        model.layers[4].kernel_regularizer.set_l1_l2(0, 0.01)
        model.layers[4].kernel_regularizer.set_prior(model.layers[4].get_weights()[0])
        model.layers[5].kernel_regularizer.set_l1_l2(0, 0.01)
        model.layers[5].kernel_regularizer.set_prior(model.layers[5].get_weights()[0])
        model.layers[7].kernel_regularizer.set_l1_l2(0, 0.01)
        model.layers[7].kernel_regularizer.set_prior(model.layers[7].get_weights()[0])
        model.layers[9].kernel_regularizer.set_l1_l2(0, 0.01)
        model.layers[9].kernel_regularizer.set_prior(model.layers[9].get_weights()[0])
        model.layers[11].kernel_regularizer.set_l1_l2(0, 0.01)
        model.layers[11].kernel_regularizer.set_prior(model.layers[11].get_weights()[0])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        with open("%s.maps" % args.model, encoding="utf-8") as f:
            word_to_idx, tgt_to_idx, max_len_lw, max_len_rw, max_len_lc, max_len_rc, chars = json.load(f)
        train_lw, train_rw, train_lc, train_rc, train_tgt, _, _ = read_dataset(args.FILE)
        train_lw = [vectorize_words(lw, word_to_idx) for lw in train_lw]
        train_rw = [vectorize_words(rw, word_to_idx, reverse=True) for rw in train_rw]
        train_lc = [vectorize_characters(lc) for lc in train_lc]
        train_rc = [vectorize_characters(rc, reverse=True) for rc in train_rc]
        train_tgt = vectorize_words(train_tgt, tgt_to_idx)
        targets = keras.utils.to_categorical(train_tgt, num_classes=len(tgt_to_idx.values()))
        train_left_words = keras.preprocessing.sequence.pad_sequences(train_lw, maxlen=max_len_lw, padding="pre", truncating="pre")
        train_right_words = keras.preprocessing.sequence.pad_sequences(train_rw, maxlen=max_len_rw, padding="pre", truncating="pre")
        train_left_chars = keras.preprocessing.sequence.pad_sequences(train_lc, maxlen=max_len_lc, padding="pre", truncating="pre")
        train_right_chars = keras.preprocessing.sequence.pad_sequences(train_rc, maxlen=max_len_rc, padding="pre", truncating="pre")
        val_lw, val_rw, val_lc, val_rc, val_tgt, _, _ = read_dataset(args.val)
        val_lw = [vectorize_words(lw, word_to_idx) for lw in val_lw]
        val_rw = [vectorize_words(rw, word_to_idx, reverse=True) for rw in val_rw]
        val_lc = [vectorize_characters(lc) for lc in val_lc]
        val_rc = [vectorize_characters(rc, reverse=True) for rc in val_rc]
        val_tgt = vectorize_words(val_tgt, tgt_to_idx)
        val_targets = keras.utils.to_categorical(val_tgt, num_classes=len(tgt_to_idx.values()))
        val_left_words = keras.preprocessing.sequence.pad_sequences(val_lw, maxlen=max_len_lw, padding="pre", truncating="pre")
        val_right_words = keras.preprocessing.sequence.pad_sequences(val_rw, maxlen=max_len_rw, padding="pre", truncating="pre")
        val_left_chars = keras.preprocessing.sequence.pad_sequences(val_lc, maxlen=max_len_lc, padding="pre", truncating="pre")
        val_right_chars = keras.preprocessing.sequence.pad_sequences(val_rc, maxlen=max_len_rc, padding="pre", truncating="pre")
        if chars:
            model.fit([train_left_words, train_right_words, train_left_chars, train_right_chars], targets, batch_size=BATCH_SIZE, epochs=args.epochs,
                      validation_data=([val_left_words, val_right_words, val_left_chars, val_right_chars], val_targets))
        else:
            model.fit([train_left_words, train_right_words], targets, batch_size=BATCH_SIZE, epochs=args.epochs,
                      validation_data=([val_left_words, val_right_words], val_targets))
        model.save("%s_adapt.h5" % args.model)
        with open("%s_adapt.maps" % args.model, mode="w", encoding="utf-8") as f:
            json.dump((word_to_idx, tgt_to_idx, max_len_lw, max_len_rw, max_len_lc, max_len_rc, chars), f, ensure_ascii=False)


def main():
    args = arguments()
    args.func(args)


if __name__ == "__main__":
    main()
