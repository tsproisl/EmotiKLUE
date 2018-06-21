#!/usr/bin/env python3

import argparse
import collections
import os.path
import re

import sklearn.naive_bayes
import sklearn.feature_extraction
import sklearn.svm
import sklearn.preprocessing

import evaluate_iest


def arguments():
    parser = argparse.ArgumentParser("Some baselines for IEST")
    parser.add_argument("-d", "--datadir", type=os.path.abspath, default="/ccl/projects/IEST/", help="Path to training and trial data; default: /ccl/projects/IEST/")
    return parser.parse_args()


def read_train_data(filename):
    data, labels = [], []
    with open(filename, encoding="utf8") as fh:
        for line in fh:
            l, d = line.strip().split("\t")
            d = re.sub(r"\[#TRIGGERWORD#\]", "", d)
            text = " ".join(d.split())
            data.append(text)
            labels.append(l)
    return data, labels


def read_labels(filename):
    with open(filename, encoding="utf8") as fh:
        labels = [l.strip() for l in fh]
    return labels


def majority_baseline(train_labels, test_labels):
    train_freq = collections.Counter(train_labels)
    most_common = train_freq.most_common(1)[0][0]
    fake_prediction = [most_common] * len(test_labels)
    print("\n## Majority baseline ##\n")
    evaluate_iest.calculatePRF(test_labels, fake_prediction)


def strip_triggerword(data):
    return [re.sub(r"\[#TRIGGERWORD#\]", "", t) for t in data]


def bow_baseline_naive_bayes(train_data, train_labels, test_data, test_labels):
    clf = sklearn.naive_bayes.MultinomialNB()
    cv = sklearn.feature_extraction.text.CountVectorizer()
    train_data = strip_triggerword(train_data)
    test_data = strip_triggerword(test_data)
    train = cv.fit_transform(train_data)
    test = cv.transform(test_data)
    clf.fit(train, train_labels)
    pred = clf.predict(test)
    print("\n## Bag of words (Naive Bayes) ##\n")
    evaluate_iest.calculatePRF(test_labels, pred.tolist())


def tfidf_baseline_naive_bayes(train_data, train_labels, test_data, test_labels):
    clf = sklearn.naive_bayes.MultinomialNB()
    cv = sklearn.feature_extraction.text.TfidfVectorizer()
    # train_data = strip_triggerword(train_data)
    # test_data = strip_triggerword(test_data)
    train = cv.fit_transform(train_data)
    test = cv.transform(test_data)
    clf.fit(train, train_labels)
    pred = clf.predict(test)
    print("\n## Bag of words tf-idf (Naive Bayes) ##\n")
    evaluate_iest.calculatePRF(test_labels, pred.tolist())


def bow_baseline_svm(train_data, train_labels, test_data, test_labels):
    cv = sklearn.feature_extraction.text.CountVectorizer()
    clf = sklearn.svm.LinearSVC()
    # scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
    # train_data = strip_triggerword(train_data)
    # test_data = strip_triggerword(test_data)
    train = cv.fit_transform(train_data)
    # train = scaler.fit_transform(train)
    test = cv.transform(test_data)
    # test = scaler.transform(test)
    clf.fit(train, train_labels)
    pred = clf.predict(test)
    print("\n## Bag of words (Linear SVC) ##\n")
    evaluate_iest.calculatePRF(test_labels, pred.tolist())


def tfidf_baseline_svm(train_data, train_labels, test_data, test_labels):
    cv = sklearn.feature_extraction.text.TfidfVectorizer()
    clf = sklearn.svm.LinearSVC()
    # scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
    # train_data = strip_triggerword(train_data)
    # test_data = strip_triggerword(test_data)
    train = cv.fit_transform(train_data)
    # train = scaler.fit_transform(train)
    test = cv.transform(test_data)
    # test = scaler.transform(test)
    clf.fit(train, train_labels)
    pred = clf.predict(test)
    print("\n## Bag of words tf-idf (Linear SVC) ##\n")
    evaluate_iest.calculatePRF(test_labels, pred.tolist())


def bigrams_svm(train_data, train_labels, test_data, test_labels):
    cv = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(1, 2))
    clf = sklearn.svm.LinearSVC()
    # scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
    # train_data = strip_triggerword(train_data)
    # test_data = strip_triggerword(test_data)
    train = cv.fit_transform(train_data)
    # train = scaler.fit_transform(train)
    test = cv.transform(test_data)
    # test = scaler.transform(test)
    clf.fit(train, train_labels)
    pred = clf.predict(test)
    print("\n## Bag of uni- and bigrams (Linear SVC) ##\n")
    evaluate_iest.calculatePRF(test_labels, pred.tolist())


def tfidf_bigrams_svm(train_data, train_labels, test_data, test_labels):
    cv = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))
    clf = sklearn.svm.LinearSVC()
    # scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
    # train_data = strip_triggerword(train_data)
    # test_data = strip_triggerword(test_data)
    train = cv.fit_transform(train_data)
    # train = scaler.fit_transform(train)
    test = cv.transform(test_data)
    # test = scaler.transform(test)
    clf.fit(train, train_labels)
    pred = clf.predict(test)
    print("\n## Bag of uni- and bigrams tf-idf (Linear SVC) ##\n")
    evaluate_iest.calculatePRF(test_labels, pred.tolist())


def main():
    args = arguments()
    train_data, train_labels = read_train_data(os.path.join(args.datadir, "train-v3.csv_tokenized.txt"))
    trial_data = read_train_data(os.path.join(args.datadir, "trial-v3.csv_tokenized.txt"))[0]
    trial_labels = read_labels(os.path.join(args.datadir, "trial-v3.labels"))
    print("# Some baselines for IEST #")
    majority_baseline(train_labels, trial_labels)
    bow_baseline_naive_bayes(train_data, train_labels, trial_data, trial_labels)
    tfidf_baseline_naive_bayes(train_data, train_labels, trial_data, trial_labels)
    bow_baseline_svm(train_data, train_labels, trial_data, trial_labels)
    tfidf_baseline_svm(train_data, train_labels, trial_data, trial_labels)
    bigrams_svm(train_data, train_labels, trial_data, trial_labels)
    tfidf_bigrams_svm(train_data, train_labels, trial_data, trial_labels)


if __name__ == "__main__":
    main()
