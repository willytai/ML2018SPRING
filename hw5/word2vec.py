import numpy as np
import util as utl
import sys

from gensim.models import Word2Vec

def main():
	filename_label   = sys.argv[1]
	filename_nolabel = sys.argv[2]

	train, label     = utl.read_train(filename_label,   label=True);  print ('training samples / labels: ', len(train), '/', label.shape[0])
	train_nlabel, _  = utl.read_train(filename_nolabel, label=False); print ('training without labels:   ', len(train_nlabel))

	# first 200000 data are labeled, the rest is not
	train_all  = train + train_nlabel

	# word2vec
	dimension = 200
	model     = Word2Vec(train_all, size=dimension, min_count=3, iter=20)
	model.save('word2vec_200.model')
	

def test():
	word_model = Word2Vec.load('word2vec_200.model')
	pretrained_weights = word_model.wv.syn0
	print (pretrained_weights.shape)

if __name__ == '__main__':
	main()
	test()