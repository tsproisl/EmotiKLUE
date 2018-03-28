#!/usr/bin/env python3

import itertools
import operator
import sys

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTMTagger(nn.Module):

    def __init__(self, word_embedding_dim, char_embedding_dim, char_hidden_dim, hidden_dim, vocab_size, charset_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.char_hidden_dim = char_hidden_dim
        self.hidden_dim = hidden_dim

        # embeddings
        self.char_embeddings = nn.Embedding(charset_size, char_embedding_dim)
        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)

        # character LSTMs
        self.forward_char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim)
        self.backward_char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        # self.char_lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.lstm = nn.LSTM(word_embedding_dim + 2 * char_hidden_dim, hidden_dim // 2, bidirectional=True)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.1)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        # self.char_hidden = self.init_bi_hidden()
        self.forward_char_hidden = self.init_hidden(char_hidden_dim)
        self.backward_char_hidden = self.init_hidden(char_hidden_dim)
        self.hidden = self.init_bi_hidden(hidden_dim)

    def init_bi_hidden(self, dim):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(2, 1, dim // 2)),
                autograd.Variable(torch.zeros(2, 1, dim // 2)))

    def init_hidden(self, dim):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, dim)),
                autograd.Variable(torch.zeros(1, 1, dim)))

    def forward(self, tokens, chars, fwd_boundaries, bwd_boundaries):
        len_chars = len(chars)
        len_tokens = len(tokens)
        char_embeds = self.char_embeddings(chars)
        # print(char_embeds.size())
        # print("first embedding")
        # print(char_embeds[0])
        rev_char_embeds = char_embeds[torch.linspace(len_chars - 1, 0, len_chars).long()]
        # print("first embedding")
        # print(rev_char_embeds[-1])
        fwd_char_lstm_out, self.forward_char_hidden = self.forward_char_lstm(char_embeds.view(len_chars, 1, -1), self.forward_char_hidden)
        bwd_char_lstm_out, self.backward_char_hidden = self.backward_char_lstm(rev_char_embeds.view(len_chars, 1, -1), self.backward_char_hidden)
        # print(fwd_char_lstm_out.size())
        # print(bwd_char_lstm_out.size())
        # print(char_lstm_out)
        fwd_char_lstm_out = fwd_char_lstm_out.view(len_chars, self.char_hidden_dim)
        bwd_char_lstm_out = bwd_char_lstm_out.view(len_chars, self.char_hidden_dim)
        fwd_char = fwd_char_lstm_out[fwd_boundaries]
        bwd_char = bwd_char_lstm_out[bwd_boundaries]
        # print(fwd_char_lstm_out.size())
        # print(bwd_char_lstm_out.size())
        # print("char lstm word ends:")
        # print(fwd_char_lstm_out[word_ends])
        # print("char lstm word starts:")
        # print(bwd_char_lstm_out[word_starts])
        embeds = self.word_embeddings(tokens)
        # print(embeds.size())
        token_repr = torch.cat((embeds, fwd_char, bwd_char), dim=1)
        # print(token_repr)
        lstm_out, self.hidden = self.lstm(token_repr.view(len_tokens, 1, -1), self.hidden)
        dropout = self.dropout(lstm_out.view(len_tokens, -1))
        tag_space = self.hidden2tag(dropout)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


def prepare_input(word_seq, word_to_ix, char_to_ix):
    tokens = prepare_sequence(word_seq, word_to_ix)
    char_seq = " ".join(word_seq)
    chars = prepare_sequence(char_seq, char_to_ix)
    len_chars = len(chars)
    word_lengths = [len(w) for w in word_seq]
    fwd_boundaries = itertools.accumulate(word_lengths, operator.add)
    fwd_boundaries = [end + i - 1 for i, end in enumerate(fwd_boundaries)]
    bwd_boundaries = itertools.accumulate(word_lengths, operator.add)
    bwd_boundaries = [start + i - wl for i, (start, wl) in enumerate(zip(bwd_boundaries, word_lengths))]
    bwd_boundaries = [len_chars - 1 - start for start in bwd_boundaries]
    return tokens, chars, fwd_boundaries, bwd_boundaries


training_data = [("Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29 .".split(), "NNP NNP , CD NNS JJ , MD VB DT NN IN DT JJ NN NNP CD .".split()),
                 ("Mr. Vinken is chairman of Elsevier N.V. , the Dutch publishing group .".split(), "NNP NNP VBZ NN IN NNP NNP , DT NNP VBG NN .".split())]

word_to_ix, tag_to_ix, char_to_ix = {}, {}, {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
            for char in word:
                if char not in char_to_ix:
                    char_to_ix[char] = len(char_to_ix)
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
char_to_ix[" "] = len(char_to_ix)
ix_to_tag = {v: k for k, v in tag_to_ix.items()}
print(word_to_ix)
print(tag_to_ix)
print(char_to_ix)

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
WORD_EMBEDDING_DIM = 64
CHAR_EMBEDDING_DIM = 32
CHAR_HIDDEN_DIM = 32
HIDDEN_DIM = 128

model = LSTMTagger(WORD_EMBEDDING_DIM, CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM, HIDDEN_DIM, len(word_to_ix), len(char_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
tokens, chars, fwd_boundaries, bwd_boundaries = prepare_input(training_data[0][0], word_to_ix, char_to_ix)
tag_scores = model(tokens, chars, fwd_boundaries, bwd_boundaries)
# print(tag_scores.size())
# print(tag_scores.max(dim=1))
print([ix_to_tag[int(ix)] for ix in tag_scores.max(dim=1)[1]])

for epoch in range(20):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.forward_char_hidden = model.init_hidden(model.char_hidden_dim)
        model.backward_char_hidden = model.init_hidden(model.char_hidden_dim)
        model.hidden = model.init_bi_hidden(model.hidden_dim)

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Variables of word indices.
        tokens, chars, fwd_boundaries, bwd_boundaries = prepare_input(sentence, word_to_ix, char_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(tokens, chars, fwd_boundaries, bwd_boundaries)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
tokens, chars, fwd_boundaries, bwd_boundaries = prepare_input(training_data[0][0], word_to_ix, char_to_ix)
tag_scores = model(tokens, chars, fwd_boundaries, bwd_boundaries)
print([ix_to_tag[int(ix)] for ix in tag_scores.max(dim=1)[1]])
