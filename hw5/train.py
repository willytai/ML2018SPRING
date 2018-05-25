import numpy as np
import util as utl
import sys

from gensim.models import word2vec
from keras.layers import LSTM, Activation, Embedding, Dense, BatchNormalization, Dropout, GRU
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping


################################
## load the pretrained word2vec
################################
word_model                 = word2vec.Word2Vec.load('word2vec_200.model')
pretrained_weights         = word_model.wv.syn0
vocab_size, Embedding_size = pretrained_weights.shape
Max_seq_length             = 55

def word2idx(word, count):
	try:
		tmp = word_model.wv.vocab[word]
	except KeyError:
		return 0, count + 1
	else:
		return tmp.index, count

def idx2word(idx):
	return word_model.wv.index2word[idx]


def CreateModel():
	###########################
	## parameter specification 
	###########################
	drop_rate = 0.2


	model = Sequential()

	# RNN
	model.add(Embedding(input_dim=vocab_size,
		output_dim=Embedding_size,
		weights=[pretrained_weights],
		trainable=False))
	# model.add(GRU(units=128, return_sequences=True,   dropout=drop_rate))
	# model.add(GRU(units=32, return_sequences=True,  dropout=drop_rate))
	model.add(GRU(units=256, return_sequences=False, dropout=drop_rate))

	# DNN
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(drop_rate))
	# model.add(Dense(32, activation='relu'))
	# model.add(Dropout(drop_rate))
	# model.add(Dense(32, activation='relu'))
	# model.add(Dropout(drop_rate))
	# model.add(Dense(32, activation='relu'))
	# model.add(Dropout(drop_rate))
	model.add(Dense(1,   activation='sigmoid'))
	return model

def WordVector():
	filename_label   = sys.argv[1] # 'database/training_label.txt'
	filename_nolabel = sys.argv[2] # 'database/training_nolabel.txt'

	# read the data
	print ('Reading texts...')
	train, label     = utl.read_train(filename_label,   label=True);  print ('training samples / labels: ', len(train), '/', label.shape[0])
	train_nlabel, _  = utl.read_train(filename_nolabel, label=False); print ('training without labels:   ', len(train_nlabel))

	# first 200000 data are labeled, the rest is not
	train_all = train + train_nlabel

	
	#############################
	## max length found to be 55
	#############################
	max = -1
	for i in range(len(train_all)):
		if max < len(train_all[i]):
			max = len(train_all[i])
	print ('max length : ', max) # ; assert max == Max_seq_length, 'max length has changed'

	count   = 0
	train_x = np.zeros([len(train_all), Max_seq_length], dtype=np.int32)
	for i, sentence in enumerate(train_all):
		for t, word in enumerate(sentence):
			train_x[i, t], count = word2idx(word, count)
			# print ('\rWords Unknown: ', count, end='', flush=True)

	print ('Found {} unknown words, replacing index with \'0\''.format(count))

	return train_x, label

def get_more_data(train_nlb, predict):
	Newlabel = []
	Newtrain = []
	Nolabel  = []
	for i in range(len(predict)):
		if predict[i] > 0.8:
			Newlabel.append(1)
			Newtrain.append(train_nlb[i])
		elif predict[i] < 0.2:
			Newlabel.append(0)
			Newtrain.append(train_nlb[i])
		else:
			Nolabel.append(train_nlb[i])

	print ('{} more training data added. {} unlabeled data remains'.format(len(Newtrain), len(Nolabel)))

	return np.array(Newtrain), np.array(Newlabel), np.array(Nolabel)

def main():
	#######################
	## training parameters
	#######################
	epochs     = 10
	batch_size = 256
	self_train = 1
	opt        = 'adam'
	loss       = 'binary_crossentropy'


	train_all, label = WordVector()

	Val_x     = train_all[:1000]       # validation X
	Val_y     = label[:1000]           # validation Y
	train     = train_all[1000:200000] # labeled data
	label     = label[1000:]           # label of train
	train_nlb = train_all[200000:]     # unlabeled data
	train_all = train_all[1000:]       # exculde the validation data

	model = CreateModel()
	model.compile(loss=loss, optimizer=opt, metrics=['acc'])
	model.summary()


	###############
	## save model
	###############
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)



	# define callbacks
	filepath   = "weights.{epoch:02d}-{val_acc:.3f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
	earlystop  = EarlyStopping(monitor='val_acc', patience=3, mode='max', verbose=1)
	callbacks_list = [earlystop, checkpoint]

	# first iteration
	model.fit(train, label, validation_data=(Val_x, Val_y), epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)

	# self training
	for iteration in range(self_train):

		# predict all of the unlabeled data
		predict = (model.predict(train_nlb, verbose=1, batch_size=batch_size)).reshape(-1)
		print ('\nGetting more data...')
		Newtrain, Newlabel, train_nlb = get_more_data(train_nlb, predict)

		# concate labels
		print ('concatenating labels...\n\n')
		label = np.hstack((label, Newlabel))
		train = np.vstack((train, Newtrain))

		model.fit(train, label, validation_data=(Val_x, Val_y), epochs=5, batch_size=batch_size, callbacks=callbacks_list)


if __name__ == '__main__':
	main()
