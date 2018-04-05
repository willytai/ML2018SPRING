import numpy as n
from itertools import chain
from keras.models import load_model
import sys, math

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

def read(testfile):
	print ('parsing test data...')
	test_x = n.genfromtxt(testfile, delimiter=',', skip_header=1, usecols=chain(range(0,1), range(78,80), range(10,11), range(1,7), range(8,10), range(11,27), range(50,54),range(55,65), range(65,71), range(72,76), range(81,123)))
	test_x = preprocess(test_x, age=True, cap_gain=True, cap_loss=True, fnlwgt=True)
	return test_x

def convert(result):
	for i in range(len(result)):
		if result[i] > 0.5:
			result[i] = 1
		else:
			result[i] = 0
	# result = list(map(int, result))
	return result

def main():
	test_x = read(sys.argv[1])

	model_1  = load_model('my_model_8603.h5')
	result_1 = model_1.predict(test_x)
	result_1 = result_1.reshape(-1)
	result_1 = convert(result_1)

	model_2  = load_model('my_model_8578.h5')
	result_2 = model_2.predict(test_x)
	result_2 = result_2.reshape(-1)
	result_2 = convert(result_2)

	model_3  = load_model('my_model_8575.h5')
	result_3 = model_3.predict(test_x)
	result_3 = result_3.reshape(-1)
	result_3 = convert(result_3)

	model_4  = load_model('my_model_8566.h5')
	result_4 = model_4.predict(test_x)
	result_4 = result_4.reshape(-1)
	result_4 = convert(result_4)

	model_5  = load_model('my_model_8563.h5')
	result_5 = model_5.predict(test_x)
	result_5 = result_5.reshape(-1)
	result_5 = convert(result_5)

	result = (result_1 + result_2 + result_3 + result_4 + result_5) / 5
	result = convert(result)
	result = list(map(int, result))

	file   = open(sys.argv[2], "w+")
	file.write('id,label\n')
	for i in range(len(result)):
		file.write(str(i+1)+','+str(result[i])+'\n')

if __name__ == '__main__':
	main()
