import os
import sys
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import cPickle
import argparse
import tsne
import matplotlib.cm as cm
from utils import dict2Mat

reload(sys)
sys.setdefaultencoding('utf-8')

def plot_with_labels(low_dim_embs, labels, groups = np.zeros(1), filename='tsne.png'):
	assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
	plt.figure(figsize=(18, 18))  #in inches
	colors = cm.rainbow(np.linspace(0, 1, max(groups)+1))
	cl = colors[0]
	for i, label in enumerate(labels):
	    x, y = low_dim_embs[i,:]

	    if len(groups) > 1:
	    	plt.scatter(x, y, s = 25, color = cl)
	    	if i < len(groups)-1 and groups[i] != groups[i+1]:
	    			cl = colors[groups[i+1]]
	    			print cl
	    			print i
	    else:
	    	plt.scatter(x, y)
	    plt.annotate(label,
	                 xy=(x, y),
	                 xytext=(5, 2),
	                 textcoords='offset points',
	                 ha='right',
	                 va='bottom')

	plt.savefig(filename)



def main(arguments):
	# TODO parse arguments
	parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--word_vec_file', help="Path to word embedding file.", type = str, required=True)
	parser.add_argument('--label_file', help = "Path to word file coresponding to embedings", type = str, required=True)
	parser.add_argument('--plot_only', help = "number of examples to plot", type = int, default = 500)
	parser.add_argument('--landmark_idx_file', help = "File that saves landmark label for each word_vec", type = str, default = '')
	parser.add_argument('--plot_by_words', help = 'If it is 1 then plot based on keywords instead of plot_only', type = int, default = 1)
	args = parser.parse_args(arguments)

	# load word embeddings
	print 'Loading vectors and labels'
	w2v = cPickle.load(open(args.word_vec_file,"r"))
	final_embeddings = dict2Mat(w2v)

	# create word_dict as lables for embedings
	with open(args.label_file) as f:
		labels = [word for line in f for word in line.split()]
	if args.landmark_idx_file != '':
		landmark_idx = cPickle.load(open(args.landmark_idx_file,'r'))
	print 'T-sne calculation'
	if args.plot_by_words == 0:
		plot_only = args.plot_only
		# using tsne in sklearn for this
		tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=10000)
		low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
		plot_with_labels(low_dim_embs, labels[:plot_only])
	else:
		# set index of clusters to plot
		plot_idx = [248,50,320,132,80,32, 1300,1403, 1294,1286,832]

		# get word embeddings and labels of them in selected clusters to plot
		begin = np.zeros((1,300))
		groups = np.zeros(1)
		plot_labels = []
		group_idx = 0 # index for groups
		for item in plot_idx:
			begin = np.concatenate((begin, final_embeddings[landmark_idx == item]),axis = 0)
			groups = np.concatenate((groups, np.ones(len(final_embeddings[landmark_idx == item]))*group_idx),axis = 0)
			group_idx = group_idx+1
			word_idxes = [i for i, elem in enumerate(landmark_idx == item, 1) if elem]
			for index in word_idxes:
				plot_labels.append(labels[index-1])
		begin = begin[1:,:]
		groups = groups[1:]
		
		# Using t-sne from original author for this problem
		low_dim_embs = tsne.tsne(begin,2,80,20.0)
		plot_with_labels(low_dim_embs, plot_labels, groups)

	print 'Finished'

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
