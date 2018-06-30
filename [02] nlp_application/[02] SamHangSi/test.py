from data_process import indexToSentence, find_vocab
from model import reRNN
import tensorflow as tf
import json

if __name__ == "__main__":
    PATH = "models"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with open('vocab.json', 'r') as fp:
        vocab = json.load(fp)
    reverse_vocab = dict()
    for key, value in vocab.items():
        reverse_vocab[value] = key
    vocab_size = len(vocab)
    model = reRNN(sess=sess, name="reRNN", max_step=50, vocab_size=vocab_size)
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(PATH))

    while(True):
        chars = input('세 글자를 입력하세요: ')
        if chars == "exit":
            break
        if len(chars) != 3:
            print("세 글자를 입력해 주세요.")
            continue
        for character in chars:
            number = find_vocab(character, vocab)
            if number == "retry":
                continue
            result = model.inference([number])
            print(reverse_vocab[number] + ' ' + indexToSentence(result, reverse_vocab)[0])
        print('')