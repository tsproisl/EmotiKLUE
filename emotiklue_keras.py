#!/usr/bin/env python3

import argparse
import itertools
import json
import os
import unicodedata

import keras.layers
import keras.models
import keras.preprocessing.sequence
import keras.utils


def read_dataset(filename):
    data = []
    vocabulary, classes = set(), set()
    with open(filename, encoding="utf8") as fh:
        for line in fh:
            if "[#TRIGGERWORD#]" not in line:
                continue
            line = unicodedata.normalize("NFD", line)
            cls, text = line.strip().split("\t")
            left_str, right_str = text.strip().split("[#TRIGGERWORD#]")
            left_words = left_str.strip().split()
            right_words = right_str.strip().split()
            vocabulary.update(set(itertools.chain(left_words, right_words)))
            classes.add(cls)
            data.append((left_words, right_words, left_str, right_str, cls))
    lw, rw, lc, rc, tgt = zip(*data)
    return lw, rw, lc, rc, tgt, vocabulary, classes


def vectorize_characters(sequence, reverse=False):
    if reverse:
        return list(reversed(sequence.encode()))
    else:
        return list(sequence.encode())


def vectorize_words(sequence, mapping, reverse=False):
    if reverse:
        sequence = reversed(sequence)
    return [mapping.get(w, len(mapping) + 1) for w in sequence]


def arguments():
    parser = argparse.ArgumentParser("EmotiKLUE")
    parser.add_argument("-m", "--model", type=os.path.abspath, required=True, help="Path to model")
    parser.add_argument("--train", type=os.path.abspath, required=True, help="Dataset for training")
    parser.add_argument("--val", type=os.path.abspath, required=True, help="Dataset for validation")
    return parser.parse_args()


def main():
    WORD_EMBEDDING_DIM = 128
    CHAR_EMBEDDING_DIM = 16
    WORD_LSTM_DIM = 128
    CHAR_LSTM_DIM = 16
    DENSE_DIM = WORD_LSTM_DIM + CHAR_LSTM_DIM
    DROPOUT = 0.1
    RECURRENT_DROPOUT = 0.1
    BATCH_SIZE = 10
    EPOCHS = 10

    args = arguments()
    train_lw, train_rw, train_lc, train_rc, train_tgt, vocabulary, classes = read_dataset(args.train)
    val_lw, val_rw, val_lc, val_rc, val_tgt, _, _ = read_dataset(args.val)

    # mappings
    word_to_idx = {w: i for i, w in enumerate(sorted(vocabulary), start=1)}
    tgt_to_idx = {c: i for i, c in enumerate(sorted(classes), start=0)}
    idx_to_tgt = {i: c for c, i in tgt_to_idx.items()}

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

    # input layers
    input_lw = keras.layers.Input(shape=(train_left_words.shape[1],))
    input_rw = keras.layers.Input(shape=(train_right_words.shape[1],))
    input_lc = keras.layers.Input(shape=(train_left_chars.shape[1],))
    input_rc = keras.layers.Input(shape=(train_right_chars.shape[1],))

    # embedding layers
    embedding_lw = keras.layers.Embedding(len(vocabulary) + 2, WORD_EMBEDDING_DIM, mask_zero=True)(input_lw)
    embedding_rw = keras.layers.Embedding(len(vocabulary) + 2, WORD_EMBEDDING_DIM, mask_zero=True)(input_rw)
    embedding_lc = keras.layers.Embedding(256, CHAR_EMBEDDING_DIM, mask_zero=True)(input_lc)
    embedding_rc = keras.layers.Embedding(256, CHAR_EMBEDDING_DIM, mask_zero=True)(input_rc)

    # LSTMs
    lstm_lw = keras.layers.LSTM(WORD_LSTM_DIM, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT)(embedding_lw)
    lstm_rw = keras.layers.LSTM(WORD_LSTM_DIM, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT)(embedding_rw)
    lstm_lc = keras.layers.LSTM(CHAR_LSTM_DIM, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT)(embedding_lc)
    lstm_rc = keras.layers.LSTM(CHAR_LSTM_DIM, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT)(embedding_rc)

    # concatenate
    lstm_out = keras.layers.Concatenate(axis=1)([lstm_lw, lstm_rw, lstm_lc, lstm_rc])

    # dense layers
    dense = keras.layers.Dense(DENSE_DIM)(lstm_out)
    predictions = keras.layers.Dense(len(classes), activation="sigmoid")(dense)

    model = keras.models.Model(inputs=[input_lw, input_rw, input_lc, input_rc], outputs=predictions)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit([train_left_words, train_right_words, train_left_chars, train_right_chars], targets, batch_size=BATCH_SIZE, epochs=EPOCHS,
              validation_data=([val_left_words, val_right_words, val_left_chars, val_right_chars], val_targets))
    model.save("%s.h5" % args.model)
    with open("%s.maps" % args.model) as f:
        json.dump((word_to_idx, tgt_to_idx), f)


if __name__ == "__main__":
    main()
