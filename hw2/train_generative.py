import numpy as n
import math, sys
from itertools import chain

def read(file_x, file_y):
	print ('parsing file...')
	x_train_bin = n.genfromtxt(file_x, delimiter=',', skip_header=1, usecols=chain(range(1,7), range(8,10), range(11,43), range(50,54),range(55,65), range(65,71), range(72,76), range(81,123))).T
	x_train_gau = n.genfromtxt(file_x, delimiter=',', skip_header=1, usecols=(0)).reshape((32561, 1)).T
	y_train = n.genfromtxt(file_y).reshape((32561, 1)).T
	# print ('x_train_bin\n', x_train_bin)
	# print ('x_train_gau\n', x_train_gau)
	# print ('y_train\n', y_train)
	return x_train_bin, x_train_gau, y_train

def validation(binx, gaux, train, size):
	valid_x_b 	= binx[:, size:]
	valid_x_g 	= gaux[:, size:]
	valid_train = train[:, size:]
	binx = binx[:,:size]
	gaux = gaux[:,:size]
	train = train[:,:size]
	return binx, gaux, train, valid_x_b, valid_x_g, valid_train

def get_class_prob(y_train):
	print ('calculating the probability of the two classes...')
	P_class1 = 0.0;
	for i in range(y_train.shape[1]):
		P_class1 += y_train[0][i]
	P_class1 /= 32561
	return P_class1, 1 - P_class1

def get_bin_prob(x_train_bin, y_train, label):
	print ('calculating the bernouli distribution of the seclected features for class{}...'.format(label))
	P_features_bin = n.zeros(x_train_bin.shape[0]).reshape((x_train_bin.shape[0], 1))
	for i in range(y_train.shape[1]):
		if y_train[0][i] == label:
			P_features_bin += x_train_bin[:, i].reshape((P_features_bin.shape[0], 1))
	return P_features_bin/y_train.shape[1]

def get_gau_param(x_train_gau, y_train, label):
	print ('calculating the gaussion distribution of the seclected features for class{}...'.format(label))
	print ('	calculating mean...')
	count = 0
	M_gau = n.zeros(x_train_gau.shape[0]).reshape((x_train_gau.shape[0], 1))
	for i in range(y_train.shape[1]):
		if y_train[0][i] == label:
			M_gau += x_train_gau[:, i].reshape((M_gau.shape[0], 1))
			count += 1
	M_gau /= count

	# print ('mean:\n', M_gau)

	print ('	calculating covariance...')
	Covar = n.zeros((x_train_gau.shape[0], x_train_gau.shape[0]))
	for i in range(x_train_gau.shape[1]):
		if y_train[0][i] == label:
			Covar += n.dot((x_train_gau[:, i].reshape((M_gau.shape[0], 1)) - M_gau), (x_train_gau[:, i].reshape((M_gau.shape[0], 1)) - M_gau).T)
	Covar /= count
	return M_gau, Covar

def main():
	filename_x_train    = sys.argv[1]
	filename_y_train    = sys.argv[2]

	x_train_bin, x_train_gau, y_train = read(filename_x_train, filename_y_train)

	x_train_bin, x_train_gau, y_train, x_train_bin_v, x_train_gau_v, y_train_v = validation(x_train_bin, x_train_gau, y_train, size=20000)


	P_class1 , P_class0 = get_class_prob(y_train)

	P_features_bin_0    = get_bin_prob(x_train_bin, y_train, 0)
	P_features_bin_1    = get_bin_prob(x_train_bin, y_train, 1)

	M_features_gau_0, Covar_0 = get_gau_param(x_train_gau, y_train, 0)
	M_features_gau_1, Covar_1 = get_gau_param(x_train_gau, y_train, 1)
	Covar                     = P_class0*Covar_0 + P_class1*Covar_1

	n.savetxt('covariance', Covar, delimiter=',')
	n.savetxt('mean_0', M_features_gau_0, delimiter=',')
	n.savetxt('mean_1', M_features_gau_1, delimiter=',')

	n.savetxt('P_class0', n.array([P_class0]))
	n.savetxt('P_class1', n.array([P_class1]))
	n.savetxt('P_features_bin_0', P_features_bin_0, delimiter=',')
	n.savetxt('P_features_bin_1', P_features_bin_1, delimiter=',')
	n.savetxt('valid_gau_x', x_train_gau_v, delimiter=',')
	n.savetxt('valid_x', x_train_bin_v, delimiter=',')
	n.savetxt('valid_y', y_train_v, delimiter=',')

	# print ('P_class0', P_class0)
	# print ('P_class1', P_class1)
	# print (P_features_bin_0)
	# print (P_features_bin_1)
	# print (x_train_bin)
	# print (x_train_bin.shape)
	# print (x_train_gau)
	# print (x_train_gau.shape)


	# print (y_train)
	# print (y_train.shape)
	# print (Covar)
	# print ('prob of income <= 50k:', P_class0)
	# print ('prob of income > 50k: ', P_class1)

if __name__ == "__main__":
	main()