from model import seq2seq
from data_process import sentence_to_char_index
import tensorflow as tf
import json

if __name__ == "__main__":
    PATH = "models"

    # load vocab, reverse_vocab, vocab_size
    with open('vocab.json', 'r') as fp:
        vocab = json.load(fp)
    reverse_vocab = dict()
    for key, value in vocab.items():
        reverse_vocab[value] = key
    vocab_size = len(vocab)

    # open session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # make model instance
    model = seq2seq(sess, encoder_vocab_size=vocab_size, decoder_vocab_size=vocab_size, max_step=50)

    # load trained model
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(PATH))

    # inference
    while True:
        test = input("User >> ")
        if test == "exit":
            break
        speak = sentence_to_char_index([test], vocab, False)
        result = model.inference([speak])
        for sentence in result:
            response = ''
            for index in sentence:
                if index == 0:
                    break
                response += reverse_vocab[index]
            print("Bot >> ", response, "\n")
