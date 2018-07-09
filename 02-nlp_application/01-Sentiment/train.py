import numpy as np
import pandas as pd
import tensorflow as tf
import os, json
from data_process import build_vocab_morphs, sentence_to_index_morphs, batch_iter
from word2vec import make_embedding_vectors
from model import CNN

if __name__ == "__main__":
    DIR = "models"

    # prepare dataset
    train = pd.read_csv('./data/train.txt', delimiter='\t')
    test = pd.read_csv('./data/test.txt', delimiter='\t')
    data = train.append(test)
    x_input = data.document
    y_input = data.label
    max_length = 30
    print('데이터로부터 정보를 얻는 중입니다.')
    embedding, vocab, vocab_size = make_embedding_vectors(list(x_input))
    print('완료되었습니다.')

    # save vocab, vocab_size, max_length
    with open('vocab.json', 'w') as fp:
        json.dump(vocab, fp)

    with open('config.txt', 'w') as f:
        f.write(str(vocab_size) + '\n')
        f.write(str(max_length))

    # import model
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


    model = CNN(sess=sess, vocab_size=vocab_size, sequence_length=max_length, trainable=True)
    model.embedding_assign(embedding)
    batches = batch_iter(list(zip(x_input, y_input)), batch_size=64, num_epochs=5)
    saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=0.5)

    # train model
    print('모델 훈련을 시작합니다.')
    avgLoss = []
    for step, batch in enumerate(batches):
        x_train, y_train = zip(*batch)
        x_train = sentence_to_index_morphs(x_train, vocab, max_length)
        l, _ = model.train(x_train, y_train)
        avgLoss.append(l)
        if step % 500 == 0:
            print('batch:', '%04d' % step, 'loss:', '%05f' % np.mean(avgLoss))
            saver.save(sess, os.path.join(DIR, "model"), global_step=step)
            avgLoss = []