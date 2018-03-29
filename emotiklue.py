#!/usr/bin/env python3

import argparse
import itertools
import os
import shutil
import time
import unicodedata

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

# https://github.com/ngarneau/understanding-pytorch-batching-lstm/blob/master/Understanding%20Pytorch%20Batching.ipynb


class IESTDataset(torch.utils.data.Dataset):
    """IEST dataset"""
    def __init__(self, filename, vectorize=None):
        self.vectorize = vectorize
        data = []
        vocabulary, classes = set(), set()
        with open(filename, encoding="utf8") as fh:
            for line in fh:
                line = unicodedata.normalize("NFC", line)
                cls, text = line.strip().split("\t")
                left_str, right_str = text.strip().split("[#TRIGGERWORD#]")
                left_words = left_str.strip().split()
                right_words = right_str.strip().split()
                vocabulary.update(set(itertools.chain(left_words, right_words)))
                classes.add(cls)
                data.append((left_words, right_words, left_str, right_str, cls))
        if self.vectorize is None:
            self.vectorize = Vectorizer(vocabulary, classes)
        data = [self.vectorize(sample) for sample in data]
        lw, rw, lc, rc, tgt = zip(*data)
        lw_lengths, rw_lengths, lc_lengths, rc_lengths = [torch.LongTensor([len(s) for s in x]) for x in (lw, rw, lc, rc)]
        self.lw_tensor = pad_sequences(lw, lw_lengths)
        self.rw_tensor = pad_sequences(rw, rw_lengths)
        self.lc_tensor = pad_sequences(lc, lc_lengths)
        self.rc_tensor = pad_sequences(rc, rc_lengths)
        self.cls_tensor = torch.LongTensor(tgt)

    def __len__(self):
        return self.cls_tensor.size(0)

    def __getitem__(self, idx):
        return self.lw_tensor[idx], self.rw_tensor[idx], self.lc_tensor[idx], self.rc_tensor[idx], self.cls_tensor[idx]


class Vectorizer(object):
    """Convert sample to Tensors."""
    def __init__(self, vocabulary, classes):
        self.word_to_ix = {w: i for i, w in enumerate(sorted(vocabulary))}
        self.cls_to_ix = {c: i for i, c in enumerate(sorted(classes))}
        self.ix_to_cls = {i: c for c, i in self.cls_to_ix.items()}

    def __call__(self, sample):
        left_words, right_words, left_str, right_str, cls = sample
        # left_tokens = autograd.Variable(torch.LongTensor([self.word_to_ix[w] for w in left_words]))
        # right_tokens = autograd.Variable(torch.LongTensor([self.word_to_ix[w] for w in right_words]))
        # left_chars = autograd.Variable(torch.LongTensor(list(left_str.encode())))
        # right_chars = autograd.Variable(torch.LongTensor(list(reversed(right_str.encode()))))
        # target = autograd.Variable(torch.LongTensor(self.cls_to_ix[cls]))

        # left_tokens = torch.LongTensor([self.word_to_ix[w] for w in left_words])
        # right_tokens = torch.LongTensor([self.word_to_ix[w] for w in right_words])
        # left_chars = torch.LongTensor(list(left_str.encode()))
        # right_chars = torch.LongTensor(list(reversed(right_str.encode())))
        # target = torch.LongTensor(self.cls_to_ix[cls])

        left_tokens = [self.word_to_ix[w] for w in left_words]
        right_tokens = [self.word_to_ix[w] for w in right_words]
        left_chars = list(left_str.encode())
        right_chars = list(reversed(right_str.encode()))
        target = self.cls_to_ix[cls]
        return left_tokens, right_tokens, left_chars, right_chars, target


class AverageMeter(object):
    """Computes and stores the average and current value

    Stolen from:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py

    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LSTMClassifier(nn.Module):
    def __init__(self, word_embedding_dim, char_embedding_dim, word_hidden_dim, char_hidden_dim, vocab_size, charset_size, class_size):
        super(LSTMClassifier, self).__init__()

        self.char_hidden_dim = char_hidden_dim
        self.word_hidden_dim = word_hidden_dim

        # embeddings
        self.char_embeddings = nn.Embedding(charset_size, char_embedding_dim)
        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)

        # character LSTMs
        self.left_char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim)
        self.right_char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim)

        # word LSTMs
        self.left_word_lstm = nn.LSTM(word_embedding_dim, word_hidden_dim)
        self.right_word_lstm = nn.LSTM(word_embedding_dim, word_hidden_dim)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.1)

        # The linear layer that maps from hidden state space to class space
        self.hidden2class = nn.Linear(2 * word_hidden_dim + 2 * char_hidden_dim, class_size)

        self.left_char_hidden = self.init_hidden(char_hidden_dim)
        self.right_char_hidden = self.init_hidden(char_hidden_dim)
        self.left_word_hidden = self.init_hidden(word_hidden_dim)
        self.right_word_hidden = self.init_hidden(word_hidden_dim)

    def init_hidden(self, dim):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, dim)),
                autograd.Variable(torch.zeros(1, 1, dim)))

    def forward(self, left_words, right_words, left_chars, right_chars):
        left_len_chars = len(left_chars)
        left_len_words = len(left_words)
        left_char_embeds = self.char_embeddings(left_chars)
        left_word_embeds = self.word_embeddings(left_words)
        left_char_lstm_out, self.left_char_hidden = self.left_char_lstm(left_char_embeds.view(left_len_chars, 1, -1), self.left_char_hidden)
        left_word_lstm_out, self.left_word_hidden = self.left_word_lstm(left_word_embeds.view(left_len_words, 1, -1), self.left_word_hidden)

        right_len_chars = len(right_chars)
        right_len_words = len(right_words)
        right_char_embeds = self.char_embeddings(right_chars)
        right_word_embeds = self.word_embeddings(right_words)
        right_char_lstm_out, self.right_char_hidden = self.right_char_lstm(right_char_embeds.view(right_len_chars, 1, -1), self.right_char_hidden)
        right_word_lstm_out, self.right_word_hidden = self.right_word_lstm(right_word_embeds.view(right_len_words, 1, -1), self.right_word_hidden)

        context = torch.cat((left_word_lstm_out[-1], right_word_lstm_out[-1], left_char_lstm_out[-1], right_char_lstm_out[-1]), dim=1)
        dropout = self.dropout(context)
        class_space = self.hidden2class(dropout)
        class_scores = F.log_softmax(class_space, dim=1)
        return class_scores


def arguments():
    parser = argparse.ArgumentParser("EmotiKLUE")
    parser.add_argument("--train", type=os.path.abspath, required=True, help="Dataset for training")
    parser.add_argument("--val", type=os.path.abspath, required=True, help="Dataset for validation")
    return parser.parse_args()


def save_checkpoint(state, is_best, filename="checkpoint.pt"):
    """Save a checkpoint that can be loaded later on to resume training.

    Stolen from:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py

    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "best_model.pt")


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k

    Stolen from:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py

    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def pad_sequences(seqs, seq_lengths):
    """Pad sequences with zeros

    Stolen from:
    https://github.com/ngarneau/understanding-pytorch-batching-lstm/blob/master/Understanding%20Pytorch%20Batching.ipynb

    """
    seq_tensor = torch.zeros((len(seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    return seq_tensor


def train(train_loader, model, loss_function, optimizer, epoch, print_freq=1):
    """Train for one epoch"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    timestamp = time.time()
    for i, (left_words, right_words, left_chars, right_chars, cls) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - timestamp)

        print("left_words:", left_words)

        left_words = autograd.Variable(left_words)
        right_words = autograd.Variable(right_words)
        left_chars = autograd.Variable(left_chars)
        right_chars = autograd.Variable(right_chars)
        cls = autograd.Variable(cls)

        print("left_words.size():", left_words.size())
        # left_words.size() == batch size?

        # Pytorch accumulates gradients. We need to clear them out
        # before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden states of the LSTMs,
        # detaching them from their history on the last instance.
        model.left_char_hidden = model.init_hidden(model.char_hidden_dim)
        model.right_char_hidden = model.init_hidden(model.char_hidden_dim)
        model.left_word_hidden = model.init_hidden(model.word_hidden_dim)
        model.right_word_hidden = model.init_hidden(model.word_hidden_dim)

        # Run forward pass and compute loss
        cls_scores = model(left_words, right_words, left_chars, right_chars)
        loss = loss_function(cls_scores, cls)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(cls_scores.data, cls, topk=(1, 3))
        losses.update(loss.data[0], left_words.size(0))
        top1.update(prec1[0], left_words.size(0))
        top3.update(prec3[0], left_words.size(0))

        # Compute gradientsa and update the parameters by calling
        # optimizer.step()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - timestamp)
        timestamp = time.time()

        if i % print_freq == 0:
            print("Epoch: [{0}][{1}/{2}]\t"
                  "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                  "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                  "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                  "Prec@3 {top3.val:.3f} ({top3.avg:.3f})".format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top3=top3))


def validate(val_loader, model, loss_function, print_freq=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    timestamp = time.time()
    for i, (left_words, right_words, left_chars, right_chars, cls) in enumerate(val_loader):

        # compute output and loss
        cls_scores = model(left_words, right_words, left_chars, right_chars)
        loss = loss_function(cls_scores, cls)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(cls_scores.data, cls, topk=(1, 3))
        losses.update(loss.data[0], left_words.size(0))
        top1.update(prec1[0], left_words.size(0))
        top3.update(prec3[0], left_words.size(0))

        # measure elapsed time
        batch_time.update(time.time() - timestamp)
        timestamp = time.time()

        if i % print_freq == 0:
            print("Test: [{0}/{1}]\t"
                  "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                  "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                  "Prec@3 {top3.val:.3f} ({top5.avg:.3f})".format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top3=top3))

    print(" * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}".format(top1=top1, top3=top3))

    return top1.avg


def main():
    WORD_EMBEDDING_DIM = 64
    CHAR_EMBEDDING_DIM = 16
    WORD_HIDDEN_DIM = 128
    CHAR_HIDDEN_DIM = 32
    CHARSET_SIZE = 256
    BATCH_SIZE = 1
    EPOCHS = 20

    args = arguments()

    # train and validation datasets
    train_dataset = IESTDataset(args.train)
    print(len(train_dataset))
    vectorize = train_dataset.vectorize
    val_dataset = IESTDataset(args.val, vectorize=vectorize)
    # CUDA: pin_memory=True?
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMClassifier(WORD_EMBEDDING_DIM, CHAR_EMBEDDING_DIM, WORD_HIDDEN_DIM, CHAR_HIDDEN_DIM, len(transform.word_to_ix), CHARSET_SIZE, len(transform.cls_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    best_prec1 = 0

    for epoch in range(EPOCHS):
        print(epoch)
        train(train_loader, model, loss_function, optimizer, epoch)
        prec1 = validate(val_loader, model, loss_function)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "best_prec1": best_prec1,
            "optimizer": optimizer.state_dict(),
        }, is_best)


if __name__ == "__main__":
    main()


# def prepare_sequence(seq, to_ix):
#     idxs = [to_ix[w] for w in seq]
#     tensor = torch.LongTensor(idxs)
#     return autograd.Variable(tensor)


# def prepare_input(left_words, right_words, left_str, right_str, word_to_ix):
#     left_tokens = prepare_sequence(left_words, word_to_ix)
#     right_tokens = prepare_sequence(reversed(right_words), word_to_ix)
#     left_chars = autograd.Variable(torch.LongTensor(list(left_str.encode())))
#     right_chars = autograd.Variable(torch.LongTensor(list(reversed(right_str.encode()))))
#     return left_tokens, right_tokens, left_chars, right_chars


# def create_indexes(training_data):
#     word_to_ix, cls_to_ix, = {}, {}
#     for left, right, l, r, cls in training_data:
#         if cls not in cls_to_ix:
#             cls_to_ix[cls] = len(cls_to_ix)
#         for word in itertools.chain(left, right):
#             if word not in word_to_ix:
#                 word_to_ix[word] = len(word_to_ix)
#     ix_to_cls = {v: k for k, v in cls_to_ix.items()}
#     return word_to_ix, cls_to_ix, ix_to_cls
