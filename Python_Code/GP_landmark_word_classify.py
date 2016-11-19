from sklearn.neighbors import NearestNeighbors
import numpy as np
import cPickle
from utils import dict2Mat
'''
This script is to classify each word vector into a class
and TODO give a scale to each word

'''

def decay_function(x,method,base, nita = 1):
	if method == 'expo':
		return base^x
	if method == 'sigm':
		return 2/(1+nita*exp(x))

if __name__=="__main__":
	# load landmarks and data
	print 'Loading data'
	landmarks = cPickle.load(open("landmarks.p","r"))
	w2v = cPickle.load(open("word_vec_dict.p","r"))
	# find nearest neighbors for landmark and save it
	W = dict2Mat(w2v)
	print 'Finding landmarks'
	nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(W)
	distances, indices = nbrs.kneighbors(landmarks)
	indices = np.reshape(indices,(len(indices)))
   	# save landmarking words
   	f_landmark = open("keyword.txt", 'w')
   	for i in range(len(indices)):
   		print>>f_landmark, w2v.items()[indices[i]][0]
   	f_landmark.close()
	print 'center word saved'
   	# give landmark label for each word
   	nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(landmarks)
   	distances_2, indices_2 = nbrs.kneighbors(landmarks)
   	# TODO use distance as a measure to mark how certain we are about this landmark.
