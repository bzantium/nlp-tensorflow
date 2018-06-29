import numpy as np
import re, random
from collections import Counter

def read_txt(data):
    lines = []
    with open(data, encoding='utf-16') as f:
        for line in f:
            lines.append(line)
    return lines

def preprocess(data):
    lines = []
    for line in data:
        line = re.sub('<head>', '', line)
        line = re.sub('</head>', '', line)
        line = re.sub('<p>', '', line)
        line = re.sub('</p>', '', line)
        line = re.sub('\n', '', line)
        line = re.sub('\"', '', line)
        lines += line.split('. ')
    return lines

def build_vocab(data):
    word_counter = Counter()
    vocab = dict()
    reverse_vocab = dict()
    for line in data:
        word_counter.update(line.split())
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    vocab_idx = 2
    for key, value in word_counter.most_common(len(word_counter)):
        vocab[key] = vocab_idx
        vocab_idx += 1
    for key, value in vocab.items():
        reverse_vocab[value] = key

    vocab_size = len(vocab)

    return vocab, reverse_vocab, vocab_size


def sentenceToIndex(lines, vocab):
    maxLength = 0
    inputSet = []
    targetSet = []
    data = []

    if len(lines) == 1:
        line = lines[0]
        line = re.sub('<head>', '', line)
        line = re.sub('</head>', '', line)
        line = re.sub('<p>', '', line)
        line = re.sub('</p>', '', line)
        line = re.sub('\n', '', line)
        line = re.sub('\"', '', line)
        line = re.sub('\.', '', line)
        data = line.split(' ')
        indexes = []
        for word in data:
            if word in vocab.keys():
                indexes.append(vocab[word])
            else:
                indexes.append(vocab['<UNK>'])
        indexes.append(0)
        inputSet = [indexes[:-1]]
        targetSet = [indexes[1:]]

    else:
        for line in lines:
            line = re.sub('<head>', '', line)
            line = re.sub('</head>', '', line)
            line = re.sub('<p>', '', line)
            line = re.sub('</p>', '', line)
            line = re.sub('\n', '', line)
            line = re.sub('\"', '', line)
            line = re.sub('\.', '', line)
            data.append(line.split(' '))

        for line in data:
            if maxLength < len(line):
                maxLength = len(line)

        for words in data:
            indexes = []
            for word in words:
                if word in vocab.keys():
                    indexes.append(vocab[word])
                else:
                    indexes.append(vocab['<UNK>'])
            for i in range(len(words), maxLength + 1):
                indexes.append(0)
            inputSet.append(indexes[:-1])
            targetSet.append(indexes[1:])

    return inputSet, targetSet

def indexToSentence(lines, reverse_vocab):
    sentences = []
    if len(lines) == 1:
        line = lines[0]
        sentence = ''
        for index in line:
            if index == 0:
                sentence = sentence[:-1]
                break
            if index == 1:
                continue
            sentence += reverse_vocab[index] + ' '
        sentences.append(sentence)

    else:
        for line in lines:
            sentence = ''
            for index in line:
                if index == 0:
                    sentence = sentence[:-1]
                    break
                if index == 1:
                    continue
                sentence += reverse_vocab[index] + ' '
            sentences.append(sentence)
    return sentences

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

def find_vocab(character, vocab):
    candidate = []
    if character == "리" or character == "니":
        character = "이"
    elif character == "림" or character == "님":
        character = "임"
    elif character == "린" or character == "닌":
        character = "인"
    elif character == "랑" or character == "낭":
        character = "앙"
    elif character == "름" or character == "늠":
        character = "음"
    elif character == "랴" or character == "냐":
        character = "야"
    elif character == "력" or character == "녁":
        character = "역"
    elif character == "류":
        character = "유"
    elif character == "로":
        character = "노"
    elif character == "려":
        character = "여"
    for key, value in vocab.items():
        if character == key[0]:
            candidate.append(value)
    try:
        result = random.sample(candidate, 1)[0]
    except ValueError:
        print("다른 글자를 입력해 주세요.")
        return "retry"
        #return random.sample(range(2,len(vocab)), 1)[0]
    return result