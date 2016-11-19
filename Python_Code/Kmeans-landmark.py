import sys
import argparse
import numpy as np
import cPickle
from utils import dict2Mat, load_bin_vec, load_words
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


"""
	This script will cluster word-embeddings into n clusters(default n=1000).
	Due to the charactor of embeddings in word2vec, each cluster will be words that have similar meanings
	It will save labels of words into idx_save_file,
"""

def find_keyword(args):
    print "loading data"
    if args.word_vec_file == '':
        w2v_file = 'GoogleNews-vectors-negative300.bin'
        keywords = load_words()
        vocab = keywords
        w2v = load_bin_vec(w2v_file, vocab)
    else :
        w2v = cPickle.load(open(args.word_vec_file,"r"))
    print "finish loading data"
    W = dict2Mat(w2v)
    kmeans = KMeans(n_clusters=args.keyword_num, random_state=0).fit(W)
    # save index to file
    cPickle.dump(kmeans.labels_,open(args.idx_save_file,"wb"))
    # get center vectors
    ctr_vecs = np.zeros(shape = (args.keyword_num,W.shape[1]))
    for i in range(args.keyword_num):
        ctr_vecs[i] = np.mean(W[kmeans.labels_==i],axis = 0)
    cPickle.dump(ctr_vecs, open('test.p',"wb"))
    print "center vecters saved"
    # save center words
    # get index of the closest vector to center vectors
    nbrs = NearestNeighbors(n_neighbors=1, algorithm=args.tree_algo).fit(W)
    distances, indices = nbrs.kneighbors(ctr_vecs)
    indices = np.reshape(indices,(len(indices)))
    # print words to file
    f_landmark = open(args.word_save_file, 'w')
    for i in range(args.keyword_num):
        print>>f_landmark, w2v.items()[indices[i]][0]
    f_landmark.close()
    print 'landmark words saved'
    # save words for vectors in W
    f_words = open(args.dict_file,'w')
    for i in range(W.shape[0]):
        print>>f_words, w2v.items()[i][0]
    f_words.close()
    print 'words saved'
    print 'all done'

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--word_vec_file', help="Path word vector .", type = str, default = '')
    parser.add_argument('--idx_save_file', help="file to save keywords indexes.", type = str, default = 'Kmeans-word-index.p')
    parser.add_argument('--word_save_file', help="file to save keywords.", type = str, default = 'Kmeans-landmark-words.txt')
    parser.add_argument('--ctr_vec_save_file', help ="file to save center vectors for each cluster", type = str, default = 'ctr_vecs.p')
    parser.add_argument('--keyword_num', help="number of clusters to find.",type = int, default = 1000)
    parser.add_argument('--tree_algo', help="method for KNN search", type = str, default = 'ball_tree')
    parser.add_argument('--dict_file', help="words for vectors", type = str, default = 'dict.txt')
    args = parser.parse_args(arguments)
	# start finding keywords
    find_keyword(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
