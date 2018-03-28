#!/usr/bin/env python3

import itertools

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


def prepare_input(left_words, right_words, left_str, right_str, word_to_ix):
    left_tokens = prepare_sequence(left_words, word_to_ix)
    right_tokens = prepare_sequence(reversed(right_words), word_to_ix)
    left_chars = autograd.Variable(torch.LongTensor(list(left_str.encode())))
    right_chars = autograd.Variable(torch.LongTensor(list(reversed(right_str.encode()))))
    return left_tokens, right_tokens, left_chars, right_chars


def create_indexes(training_data):
    word_to_ix, cls_to_ix, = {}, {}
    for left, right, l, r, cls in training_data:
        if cls not in cls_to_ix:
            cls_to_ix[cls] = len(cls_to_ix)
        for word in itertools.chain(left, right):
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    ix_to_cls = {v: k for k, v in cls_to_ix.items()}
    return word_to_ix, cls_to_ix, ix_to_cls


def main():
    training_data = [("@USERNAME A little ", " that I am not invited for drinks anymore! :-(", "anger"),
                     ("@USERNAME @USERNAME It's pretty ", " that there would even BE stock photos for an event like this.", "disgust"),
                     ("Apparently I've been black mailing my brother since he was 8 and he will forever be ", " because of it 😑😂", "fear"),
                     ("Republicans are so ", " that people may treat people like people and not tax margins...wouldn't want to appear human...", "fear"),
                     ("Katy once felt so ", " that she barely had the strength to live another day and look at her now she's the happiest person alive", "sad"),
                     ("Number one comment on the Internet every day: \"I am very ", " because not everyone feels exactly the way that I do about everything,\"", "anger"),
                     ("@USERNAME @USERNAME I'm ever more ", " that #POTUS & #FLOTUS take it in stride. #ClassyCouple Always the higher road taken #Obama", "surprise"),
                     ("@USERNAME @USERNAME Woohoo! I'm ", " because I breastfed my babies! IN PUBLIC! Quick, lock me up!", "disgust"),
                     ("@USERNAME — pain I'm in from being so lonely. But I'm ", " that this girl before me didn't make an excuse to leave. Her motive —", "surprise"),
                     ("pretty ", " that a white republican man keeps introducing other white men on the house floor as ANOTHER CHAMPION OF LIFE. #hr7", "disgust"),
                     ("@USERNAME no cause I don't wanna be that ", " that I eat an entire thing of ice cream", "sad"),
                     ("Always be ", " when a woman hits you with the “K.\"", "fear"),
                     ("@USERNAME AHH YOU'RE SO SWEET TYSM im just ", " that we can appreciate akiham together!!!!", "joy"),
                     ("Drake wasn't ", " because Madonna kissed him, it's because she reminded him of an ex.. ahahaha", "disgust"),
                     ("\"And I'm ", " that even as we integrate, we are walking into a place that does not understand...", "fear"),
                     ("i want to make a curiouscat but too ", " that i wont get any questions and thus look like more of a loser than i already am. insecurity is pretty cool but thats what i get for attention seeking on the INTERNET HAHAH http://url.removed", "fear"),
                     ("Why is everybody so ", " that I can't scooter❓❓❓", "surprise"),
                     ("@USERNAME @USERNAME Very ", " that POTUS can't stop tweeting like a 12 year old girl long enough to do his job.", "sad"),
                     ("@USERNAME he is un", " because he is normal.[NEWLINE]His health reports are normal!", "sad"),
                     ("It's bare ", " when fat girls tweet \"eat the booty like groceries\". Can you even wash your bum properly you fat shit", "disgust")]

    # pseudo-preprocessing
    training_data = [(l.strip().split(), r.strip().split(), l, r, c) for l, r, c in training_data]

    word_to_ix, cls_to_ix, ix_to_cls = create_indexes(training_data)

    WORD_EMBEDDING_DIM = 64
    CHAR_EMBEDDING_DIM = 16
    WORD_HIDDEN_DIM = 128
    CHAR_HIDDEN_DIM = 32
    CHARSET_SIZE = 256

    model = LSTMClassifier(WORD_EMBEDDING_DIM, CHAR_EMBEDDING_DIM, WORD_HIDDEN_DIM, CHAR_HIDDEN_DIM, len(word_to_ix), CHARSET_SIZE, len(cls_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # random output before training
    left_words, right_words, left_str, right_str = training_data[0][0:4]
    left_tokens, right_tokens, left_chars, right_chars = prepare_input(left_words, right_words, left_str, right_str, word_to_ix)
    cls_scores = model(left_tokens, right_tokens, left_chars, right_chars)
    print([ix_to_cls[int(ix)] for ix in cls_scores.max(dim=1)[1]])

    for epoch in range(20):
        print(epoch)
        for left_words, right_words, left_str, right_str, cls in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Also, we need to clear out the hidden states of the LSTM,
            # detaching it from its history on the last instance.
            model.left_char_hidden = model.init_hidden(model.char_hidden_dim)
            model.right_char_hidden = model.init_hidden(model.char_hidden_dim)
            model.left_word_hidden = model.init_hidden(model.word_hidden_dim)
            model.right_word_hidden = model.init_hidden(model.word_hidden_dim)

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Variables of word indices.
            left_tokens, right_tokens, left_chars, right_chars = prepare_input(left_words, right_words, left_str, right_str, word_to_ix)
            target = autograd.Variable(torch.LongTensor([cls_to_ix[cls]]))

            # Step 3. Run our forward pass.
            cls_scores = model(left_tokens, right_tokens, left_chars, right_chars)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(cls_scores, target)
            loss.backward()
            optimizer.step()

    # we should have learned something
    left_words, right_words, left_str, right_str = training_data[0][0:4]
    left_tokens, right_tokens, left_chars, right_chars = prepare_input(left_words, right_words, left_str, right_str, word_to_ix)
    cls_scores = model(left_tokens, right_tokens, left_chars, right_chars)
    print([ix_to_cls[int(ix)] for ix in cls_scores.max(dim=1)[1]])


if __name__ == "__main__":
    main()
