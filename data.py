import torch 
import logging
#from pytorch_transformers import BertModel, OpenAIGPTModel, GPT2Model
import nltk
from tqdm import tqdm
import copy
import pickle
nltk.download('punkt')
import time
import sklearn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from gensim.models import KeyedVectors
import io


def fetch_word2vec_embeddings(vocab, pickled=False):
    print("Fetching Word2vec Embeddings")
    pickle_path = "word2vec.vocab_size={}.pickle".format(len(vocab))
    if pickled:
        output = pickle.load(open(pickle_path, 'rb'))
    else:
        filepath = "GoogleNews-vectors-negative300.bin"
        wv_from_bin = KeyedVectors.load_word2vec_format(filepath, binary=True)
        print(wv_from_bin["hi"])
        #output = {w : wv_from_bin[translation_dict.get(w, w)] for w in vocab}
        output = {w : wv_from_bin[w] for w in vocab}
        pickle.dump(output, open(pickle_path, 'wb'))
    return output


def fetch_glove_embeddings(pickled=False):
    print("Fetching GloVe Embeddings")
    dimension = 300
    print("Fetching GloVe {} Dimensional Embeddings".format(dimension))
    path = "glove.6B.{}d".format(dimension)
    pickle_path = path + ".pickle"
    if pickled:
        output = pickle.load(open(pickle_path, 'rb'))
        embedding_matrix, w2i, i2w = output
    else:
        w2i, i2w = {}, {}
        embedding_matrix = []
        with open(path + ".txt", mode = 'r', encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f)):
                tokens = line.split()
                word = tokens[0]
                embedding = np.array([float(val) for val in tokens[1:]] )
                w2i[word] = i 
                i2w[i] = word 
                embedding_matrix.append(embedding)
                assert len(embedding) == dimension
        embedding_matrix = np.array(embedding_matrix)
        output = (embedding_matrix, w2i, i2w)
        pickle.dump(output, open(pickle_path, 'wb'))
    print("Shape of GloVe Embedding Matrix: {}".format(embedding_matrix.shape))
    return output


def fetch_static_embeddings(word2vec, glove, w2i, vocab, pickled=False):
    print("Fetching Word2vec and GloVe Static Embeddings")
    pickle_f = "static_vocab_size={}.decontextualized.pickle".format(len(vocab))
    if pickled:
        score_embeddings = pickle.load(open(pickle_f, 'rb'))
    else:
        score_embeddings = {}
        for w in vocab:
            score_embeddings[w] = {}
            if w in w2i:
                score_embeddings[w]["glove"] = glove[w2i[w]]
            else:
                print("Not in GloVe:", w)
                #score_embeddings[w]["glove"] = glove[w2i[translation_dict[w]]]
                score_embeddings[w]["glove"] = glove[w2i[w]]
            if w in word2vec:
                score_embeddings[w]["word2vec"] = word2vec[w]
            else:
                print("Not in Word2vec:", w)
        pickle.dump(score_embeddings, open(pickle_f, 'wb'))
    return score_embeddings
    