from model import CNN
from data_process import sentence_to_index_morphs
import tensorflow as tf
import re, json

if __name__ == "__main__":
    PATH = "models"

    # load vocab, vocab_size, max_length
    with open('vocab.json', 'r') as fp:
        vocab = json.load(fp)

    with open('config.txt', 'r') as f:
        vocab_size = int(re.sub('\n', '', f.readline()))
        max_length = int(f.readline())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    model = CNN(sess=sess, vocab_size=vocab_size, sequence_length=max_length, trainable=True)
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(PATH))

    while True:
        test = input("User >> ")
        if test == "exit":
            break
        speak = sentence_to_index_morphs([test], vocab, max_length)
        label, prob = model.predict(speak)
        if prob[0] < 0.6:
            response = '차분해 보이시네요 :)'
        else:
            if label[0] == 0:
                response = '기분이 좋지 않아 보여요 :('
            else:
                response = '기분이 좋아 보이시네요!'
        print("Bot >> ", response, "\n")