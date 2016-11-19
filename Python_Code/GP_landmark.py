import sys
import numpy as np
from numpy import linalg as LA
import numpy as np
import cPickle
from manifold_gp import ManifoldGP
from utils import load_words,load_bin_vec, add_unknown_words,dict2Mat


""" This preprocessing script is modified based on zhe's modified version of
    https://github.com/yoonkim/CNN_sentence
    It used manifold guassian process code from Dawen Liang<dliang@ee.columbia.edu>
    https://github.com/dawenl/manifold_landmarks.git to learn landmarks.
"""



if __name__=="__main__":
    w2v_file = 'GoogleNews-vectors-negative300.bin'

    print "preparing data...\n"
    # load words in translation library and create a dictionary with word as key and embedding as value
    keywords = load_words()
    vocab = keywords
    w2v = load_bin_vec(w2v_file, vocab)
    # add_unknown_words(w2v, vocab)
    # construct matrix of word embeddings to fit into Gaussian Process.
    W = dict2Mat(w2v)
    # get landmarks
    landmark = ManifoldGP()
    landmark.learn_landmarks(W)
    # save landmarks learned by gp, save word-embedding dictionary.
    cPickle.dump(landmark.landmarks,open("GP_landmarks.p","wb"))
    print "landmarks saved!"
    cPickle.dump(w2v,open("word_vec_dict.p","wb"))
    print "dictionary saved!"
