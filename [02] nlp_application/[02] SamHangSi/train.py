import numpy as np
import tensorflow as tf
from model import reRNN
from data_process import read_txt, preprocess, build_vocab, batch_iter, sentenceToIndex
import os, json

if __name__ == "__main__":
    PATH = 'models'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True    
    sess = tf.Session(config=config)
    data = read_txt('./data/novel.txt')
    data = preprocess(data)
    vocab, reverse_vocab, vocab_size = build_vocab(data)
    with open('vocab.json', 'w') as fp:
        json.dump(vocab, fp)
    model = reRNN(sess=sess, name="reRNN", max_step=20, vocab_size=vocab_size)
    batches = batch_iter(data, batch_size=64, num_epochs=1001)
    saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=0.5)

    print('모델 훈련을 시작합니다.')
    avgLoss = []
    for step, batch in enumerate(batches):
        x_train, y_train = sentenceToIndex(batch, vocab)
        l, _ = model.train(x_train, y_train)
        avgLoss.append(l)
        if step % 500 == 0:
            print('batch:', '%04d' % step, 'loss:', '%05f' % np.mean(avgLoss))
            saver.save(sess, os.path.join(PATH, 'my-model.ckpt'), global_step=step)
            avgLoss = []