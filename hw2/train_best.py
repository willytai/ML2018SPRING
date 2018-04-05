import numpy as n
import math, sys
from itertools import chain

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU

def normalize(x, mean, std):
	return (x - mean) / std

def preprocess(x_train, age=False, cap_gain=False, cap_loss=False, fnlwgt=False):
	print ('preprocessing data...')

	std  = n.std(x_train, axis=0)
	mean = n.mean(x_train, axis=0)
	max  = n.amax(x_train, axis=0)
	# print (max); sys.exit()
	for i in range(x_train.shape[0]):
		if age == True:
			x_train[i][0] = normalize(x_train[i][0], mean[0], std[0])
			# x_train[i][0] = x_train[i][0] / 25

		if cap_gain == True:
			x_train[i][1] = normalize(x_train[i][1], mean[1], std[1])
			# x_train[i][1] = x_train[i][1] / max[1]

		if cap_loss == True:
			x_train[i][2] = normalize(x_train[i][2], mean[2], std[2])
			# x_train[i][2] = x_train[i][2] / max[2]

		if fnlwgt == True:
			x_train[i][3] = normalize(x_train[i][3], mean[3], std[3])
			# x_train[i][3] = x_train[i][3] / max[3]

	return x_train


def read(file_x, file_y):
	print ('parsing file...')
	x_train = n.genfromtxt(file_x, delimiter=',', skip_header=1, usecols=chain(range(0,1), range(78,80), range(10,11), range(1,7), range(8,10), range(11,27), range(50,54),range(55,65), range(65,71), range(72,76), range(81,123)))
	y_train = n.genfromtxt(file_y).reshape((32561, 1))
	x_train = preprocess(x_train, age=True, cap_gain=True, cap_loss=True, fnlwgt=True)
	return x_train, y_train

def validation(train_x, train_y, portion):
	total 	= train_x.shape[0]

	valid_x = train_x[int(portion*total):, :,]
	valid_y	= train_y[int(portion*total):, :,]

	train_x = train_x[:int(portion*total), :,]
	train_y	= train_y[:int(portion*total), :,]

	return valid_x, valid_y, train_x, train_y

def convert(result):
	for i in range(len(result)):
		if result[i] > 0.5:
			result[i] = int(1)
		else:
			result[i] = int(0)
	result = list(map(int, result))
	return result

def main():
	filename_x_train = sys.argv[1]
	filename_y_train = sys.argv[2]

	x_train, y_train = read(filename_x_train, filename_y_train)
	valid_x, valid_y, x_train, y_train = validation(x_train, y_train, 0.9)

	model = Sequential()
	
	model.add(Dense(input_dim=x_train.shape[1], units=100, activation='sigmoid'))
	model.add(Dense(units=100, activation='relu'))
	# model.add(Dense(units=256, activation='relu'))
	# model.add(Dense(units=128, activation='relu'))
	# model.add(Dense(units=64, activation='relu'))
	# model.add(Dense(units=32, activation='relu'))
	model.add(Dense(units=1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(x_train, y_train, batch_size=16, epochs=15)

	result = model.predict(valid_x)
	result = result.reshape(-1)
	result = convert(result)
	err = 0
	for i in range(len(result)):
		if result[i] != valid_y[i][0]:
			err += 1
	print ("acc:", 1-err/len(result))

	# scores = model.evaluate(valid_x, valid_y)
	# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	model.save('my_model.h5')

if __name__ == "__main__":
	main()