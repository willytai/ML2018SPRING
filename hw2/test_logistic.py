import numpy as n
from itertools import chain
import sys, math

def normalize(x, mean, std):
	return (x - mean) / std

def preprocess(x_train, age=False, cap_gain=False, cap_loss=False, fnlwgt=False):
	print ('preprocessing data...')

	std  = n.sqrt(n.std(x_train, axis=0))
	mean = n.mean(x_train, axis=0)
	max  = n.amax(x_train, axis=0)
	# print (max); sys.exit()
	for i in range(x_train.shape[0]):
		if age == True:
			# x_train[i][0] = normalize(x_train[i][0], mean[0], std[0])
			x_train[i][0] = x_train[i][0] / 25

		if cap_gain == True:
			x_train[i][1] = normalize(x_train[i][1], mean[1], std[1])
			# x_train[i][1] = x_train[i][1] / max[1]

		if cap_loss == True:
			x_train[i][2] = normalize(x_train[i][2], mean[2], std[2])
			# x_train[i][2] = x_train[i][2] / max[2]

		if fnlwgt == True:
			# x_train[i][3] = normalize(x_train[i][3], mean[3], std[3])
			x_train[i][3] = x_train[i][3] / max[3]

	return x_train

def read(testfile):
	print ('parsing test data...')
	test_x = n.genfromtxt(testfile, delimiter=',', skip_header=1, usecols=chain(range(0,1), range(78,80), range(10,11), range(1,7), range(8,10), range(11,43), range(50,54),range(55,65), range(65,71), range(72,76), range(81,123)))
	test_x = preprocess(test_x, age=True, cap_gain=True, cap_loss=True, fnlwgt=True)
	return test_x

def sigmoid(z):
	if z > 20:
		return 1
	elif z < -20:
		return 0
	else:
		return 1 / (1 + math.exp(-z))

def f_w_b(x, w, b):
	return sigmoid(b + n.dot(x, w))

def predict(weight, bias, x):
	result = []
	for i in range(x.shape[0]):
		if x[i][0] <= 1:
			result.append(0)
		else:
			result.append(f_w_b(x[i], weight, bias))
	for i in range(len(result)):
		if (result[i] > 0.5):
			result[i] = 1
		else:
			result[i] = 0
	result = n.asarray(result)
	return result

def main():
	weight = n.loadtxt('weight.txt', delimiter=',')
	bias   = n.loadtxt('bias.txt', delimiter=',')
	test_x = read(sys.argv[1])

	result = predict(weight, bias, test_x)

	file   = open(sys.argv[2], "w+")
	file.write('id,label\n')
	for i in range(len(result)):
		file.write(str(i+1)+','+str(result[i])+'\n')

if __name__ == '__main__':
	main()