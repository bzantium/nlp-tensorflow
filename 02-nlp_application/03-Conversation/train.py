from model import seq2seq
import tensorflow as tf
import os, re, json
import numpy as np
from data_process import read_txt, build_character, make_dataset, batch_iter, sentence_to_char_index

if __name__ == "__main__":
    DIR = "models"

    # read and build dataset
    data = read_txt('./data/dialog.txt')
    vocab, reverse_vocab, vocab_size = build_character(data)

    # save vocab
    with open('vocab.json', 'w') as fp:
        json.dump(vocab, fp)

    # open session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # make model instance
    model = seq2seq(sess, encoder_vocab_size=vocab_size, decoder_vocab_size=vocab_size)

    # make train batches
    input, target = make_dataset(data)
    batches = batch_iter(list(zip(input, target)), batch_size=64, num_epochs=1001)

    # model saver
    saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=0.5)

    # train model
    print('모델 훈련을 시작합니다.')
    avgLoss = []
    for step, batch in enumerate(batches):
        x_train, y_train = zip(*batch)
        x_train = sentence_to_char_index(x_train, vocab, is_target=False)
        y_train = sentence_to_char_index(y_train, vocab, is_target=True)
        l, _ = model.train(x_train, y_train)
        avgLoss.append(l)
        if step % 100 == 0:
            print('batch:', '%04d' % step, 'loss:', '%.5f' % np.mean(avgLoss))
            saver.save(sess, os.path.join(DIR, "model"), global_step=step)
            avgLoss = []