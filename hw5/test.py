import numpy as np
import sys
import util as utl
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from gensim.models import word2vec



testing = sys.argv[1]
predict = sys.argv[2]


def read_test(filename):
	test   = []
	header = True
	with open(filename, 'r') as f:
		for line in f:

			# skip the header
			if header == True:
				header = False
				continue

			# find the first comma
			i = 0
			while line[i] != ',':
				i += 1
			
			# slice the string
			line = line[i+1:]

			test.append(utl.text_to_wordlist(line))

	return test

def to_0_1(data):
	label = []
	for i in range(len(data)):
		if data[i] >= 0.501:
			label.append(1)
		else:
			label.append(0)

	return np.array(label, dtype='float')

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# make five copy
models = model_from_json(loaded_model_json)

models.summary()

# load weights into new model
models.load_weights("weights.03-0.831.hdf5")
print("Loaded model from disk")

print ("compiling model...")
models.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

print ('reading testing data...')
test = read_test(testing)

print ('Creating sequences from Word2Vec...')

def word2idx(word):
	try:
		tmp = word_model.wv.vocab[word]
	except KeyError:
		return 0, False
	else:
		return tmp.index, True

Max_seq_length = 55
word_model     = word2vec.Word2Vec.load('word2vec_200.model')
test_x         = np.zeros([len(test), Max_seq_length], dtype=np.int32)

# skip unknown words
for i, sentence in enumerate(test):
	idx = 0
	for word in sentence:
		wdid, found = word2idx(word)
		if found == True:
			test_x[i, idx] = wdid
			idx += 1

result0 = models.predict(test_x, verbose=1, batch_size=256).reshape(-1)

result = to_0_1(result0)

print ('writing file')

file = open(predict, 'w+')
file.write('id,label\n')
for i in range(len(result)):
	file.write('{},{}\n'.format(i, int(result[i])))
file.close()
