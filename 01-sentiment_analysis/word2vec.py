import numpy as np
import gensim
from data_process import morphs_process

def make_embedding_vectors(data, embedding_size=300):
    tokens = morphs_process(data)
    wv_model = gensim.models.Word2Vec(min_count=1, window=5, size=embedding_size)
    wv_model.build_vocab(tokens)
    wv_model.train(tokens, total_examples=wv_model.corpus_count, epochs=wv_model.epochs)
    word_vectors = wv_model.wv
    
    vocab = dict()
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    idx = 2
    for word in word_vectors.vocab:
        vocab[word] = idx
        idx += 1
        
    embedding = []
    embedding.append(np.random.normal(size=300))
    embedding.append(np.random.normal(size=300))
    for word in word_vectors.vocab:
        embedding.append(word_vectors[word])
    embedding = np.asarray(embedding)
    vocab_size = len(embedding)
    
    return embedding, vocab, vocab_size