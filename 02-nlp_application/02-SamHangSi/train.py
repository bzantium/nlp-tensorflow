import numpy as np
import tensorflow as tf
from model import reRNN
from data_process import read_txt, preprocess, build_vocab, batch_iter, sentenceToIndex
import os, json

if __name__ == "__main__":
    DIR = "models"

    # read and build dataset
    data = read_txt('./data/novel.txt')
    data = preprocess(data)
    vocab, reverse_vocab, vocab_size = build_vocab(data)

    # save vocab
    with open('vocab.json', 'w') as fp:
        json.dump(vocab, fp)

    # open session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # make model instance
    model = reRNN(sess=sess, vocab_size=vocab_size, lr=1e-1)

    # make train batches
    batches = batch_iter(data, batch_size=64, num_epochs=1001)

    # model saver
    saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=0.5)

    # train model
    print('모델 훈련을 시작합니다.')
    avgLoss = []
    for step, batch in enumerate(batches):
        x_train, y_train = sentenceToIndex(batch, vocab)
        l, _ = model.train(x_train, y_train)
        avgLoss.append(l)
        if step % 100 == 0:
            print('batch:', '%04d' % step, 'loss:', '%.5f' % np.mean(avgLoss))
            saver.save(sess, os.path.join(DIR, 'my-model.ckpt'), global_step=step)
            avgLoss = []