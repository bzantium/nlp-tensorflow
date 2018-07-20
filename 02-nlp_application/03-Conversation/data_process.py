import re
from collections import Counter
import numpy as np

def read_txt(data):
    lines = []
    with open(data, encoding='utf-8') as f:
        for line in f:
            lines.append(re.sub('\n', '', line))
    return lines

def tokenizer(sentence):
    tokens = re.findall(r"[\w]+|[^\s\w]", sentence)
    return tokens


def build_character(sentences):
    word_counter = Counter()
    vocab = dict()
    reverse_vocab = dict()

    for sentence in sentences:
        tokens = list(sentence)
        word_counter.update(tokens)

    vocab['<PAD>'] = 0
    vocab['<GO>'] = 1
    vocab['<UNK>'] = 2
    vocab_idx = 3

    for key, value in word_counter.most_common(len(word_counter)):
        vocab[key] = vocab_idx
        vocab_idx += 1

    for key, value in vocab.items():
        reverse_vocab[value] = key

    vocab_size = len(vocab.keys())

    return vocab, reverse_vocab, vocab_size


def build_vocab(sentences):
    word_counter = Counter()
    vocab = dict()
    reverse_vocab = dict()

    for sentence in sentences:
        tokens = tokenizer(sentence)
        word_counter.update(tokens)

    vocab['<PAD>'] = 0
    vocab['<GO>'] = 1
    vocab['<UNK>'] = 2
    vocab_idx = 3

    for key, value in word_counter.most_common(len(word_counter)):
        vocab[key] = vocab_idx
        vocab_idx += 1

    for key, value in vocab.items():
        reverse_vocab[value] = key

    vocab_size = len(vocab.keys())

    return vocab, reverse_vocab, vocab_size


def sentence_to_char_index(lines, vocab, is_target=False):
    tokens = []
    indexes = []
    max_len = 0

    if len(lines) == 1:
        tokens = list(lines[0])
        for token in tokens:
            if token in vocab.keys():
                indexes.append(vocab[token])
            else:
                indexes.append(vocab['<UNK>'])

    else:
        for sentence in lines:
            token = list(sentence)
            tokens.append(token)
            length = len(token)
            if max_len < length:
                if is_target == True:
                    max_len = length + 1
                else:
                    max_len = length

        for token in tokens:
            temp = token
            for _ in range(len(temp), max_len):
                temp.append('<PAD>')
            index = []
            for char in temp:
                if char in vocab.keys():
                    index.append(vocab[char])
                else:
                    index.append(vocab['<UNK>'])
            indexes.append(index)

    return indexes


def sentence_to_word_index(lines, vocab, is_target=False):
    tokens = []
    indexes = []
    max_len = 0

    if type(lines) is str:
        tokens = tokenizer(lines)
        for token in tokens:
            if token in vocab.keys():
                indexes.append(vocab[token])
            else:
                indexes.append(vocab['<UNK>'])

    else:
        for sentence in lines:
            token = tokenizer(sentence)
            tokens.append(token)
            length = len(token)
            if max_len < length:
                if is_target == True:
                    max_len = length + 1
                else:
                    max_len = length

        for token in tokens:
            temp = token
            for _ in range(len(temp), max_len):
                temp.append('<PAD>')
            index = []
            for char in temp:
                if char in vocab.keys():
                    index.append(vocab[char])
                else:
                    index.append(vocab['<UNK>'])
            indexes.append(index)

    return indexes


def make_dataset(data):
    input = []
    target = []
    for i in range(len(data)-1):
        input.append(data[i])
        target.append(data[i+1])
    return input, target

	
def make_dataset_for_translation(data):
    input = []
    target = []
    for i in range(0, len(data), 2):
        input.append(data[i])
        target.append(data[i+1])
    return input, target


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]