import numpy as n
import math, sys
from itertools import chain

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


def read(file_x, file_y):
	print ('parsing file...')
	x_train = n.genfromtxt(file_x, delimiter=',', skip_header=1, usecols=chain(range(0,1), range(78,80), range(10,11), range(1,7), range(8,10), range(11,43), range(50,54),range(55,65), range(65,71), range(72,76), range(81,123)))
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

def sigmoid(z):
	if z > 50:
		return 1
	elif z < -50:
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
			result[i] = 0
		else:
			result[i] = 1
	result = n.asarray(result)
	return result

def train(train, lable, valid_x, valid_y, iterat, lr, adag=True, reg=0):
	feature_num = train.shape[1]
	data_num    = train.shape[0]

	weight = n.ones(feature_num)
	bias   = 0.0

	lr_w   = n.ones(feature_num)
	lr_b   = 0.0

	for it in range(iterat):

		b_grad = 0.0
		w_grad = n.ones(feature_num)

		for i in range(data_num):
			w_grad -= (lable[i][0] - f_w_b(train[i], weight, bias))*train[i]
			b_grad -= (lable[i][0] - f_w_b(train[i], weight, bias))

		if adag == True:
			lr_w   += w_grad ** 2
			lr_b   += b_grad ** 2

			weight -= lr/n.sqrt(lr_w) * w_grad
			bias   -= lr/n.sqrt(lr_b) * b_grad
		else:
			weight -= lr * w_grad
			bias   -= lr * b_grad

		if it % 5 == 0:
			result = predict(weight, bias, valid_x)
			err = 0.0
			for i in range(valid_y.shape[0]):
				if result[i] != valid_y[i][0]:
					err += 1
			err /= valid_y.shape[0]

			print ('accuracy after {}th iteration (testing set) ...'.format(it), err)

			result = predict(weight, bias, train)
			err = 0.0
			for i in range(lable.shape[0]):
				if result[i] != lable[i][0]:
					err += 1
			err /= lable.shape[0]

			print ('accuracy after {}th iteration (training set)...'.format(it), err)
			print ('')

	return weight, bias

def main():
	filename_x_train = sys.argv[1]
	filename_y_train = sys.argv[2]

	x_train, y_train = read(filename_x_train, filename_y_train)
	valid_x, valid_y, x_train, y_train = validation(x_train, y_train, 0.9)

	weight, bias = train(x_train, y_train, valid_x, valid_y, iterat=200, lr=0.2, adag=True)

	file = open ('bias.txt', "w+")
	file.write(str(bias))
	file.close()

	n.savetxt('weight.txt', weight, delimiter=',')


if __name__ == "__main__":
	main()
