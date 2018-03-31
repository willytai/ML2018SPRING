import numpy as n
import sys
from itertools import chain
from scipy.stats import multivariate_normal

def read(testfile):
	print ('parsing test data...')
	x_test_bin = n.genfromtxt(testfile, delimiter=',', skip_header=1, usecols=chain(range(1,7), range(8,10), range(11,43), range(50,54),range(55,65), range(65,71), range(72,76), range(81,123))).T
	x_test_gau = n.genfromtxt(testfile, delimiter=',', skip_header=1, usecols=(0)).reshape((16281, 1)).T
	return x_test_bin, x_test_gau

def dot(prob, test):
	# print ('prob\n', prob); print ('test\n', test);
	dot = 0.0
	for i in range(len(test)):
		if test[i] == 1:
			dot += prob[i][0]
		elif test[i] == 0:
			dot += (1 - prob[i][0])
		else:
			print ('not binary feature! index: ',i);
	# print (' after dot:', dot);
	return dot

def pdf_gau(x, mean, cov):
	mean = mean.reshape(-1)
	# print(x)
	y = multivariate_normal.pdf(x, mean=mean, cov=cov)
	# print (y); sys.exit()
	return y


def test_valid():
	x_valid  = n.genfromtxt('valid_x', delimiter=',')
	y_valid  = n.genfromtxt('valid_y', delimiter=',')
	P_class0 = n.loadtxt('P_class0', delimiter=',')
	P_class1 = n.loadtxt('P_class1', delimiter=',')
	p_features_bin_0 = n.loadtxt('p_features_bin_0', delimiter=',').reshape((x_valid.shape[0],1))
	p_features_bin_1 = n.loadtxt('p_features_bin_1', delimiter=',').reshape((x_valid.shape[0],1))
	Covar  = n.loadtxt('Covariance', delimiter=',')
	mean_0 = n.loadtxt('mean_0', delimiter=',').reshape((1,1))
	mean_1 = n.loadtxt('mean_1', delimiter=',').reshape((1,1))
	x_gau_valid = n.loadtxt('valid_gau_x', delimiter=',').reshape((12561, 1)).T

	# rint (x_valid)
	# print(y_valid)

	result = []
	for i in range(x_valid.shape[1]):
		prob_0 = P_class0*pdf_gau(x_gau_valid[:,i], mean_0, Covar) / (P_class0*pdf_gau(x_gau_valid[:,i], mean_0, Covar) + P_class1*pdf_gau(x_gau_valid[:,i], mean_1, Covar))
		if prob_0 > 0.5:
			result.append(0)
		else:
			result.append(1)
	err = 0
	for i in range(len(result)):
		if result[i] != y_valid[i]:
			err += 1
	print ('accuracy: ', err/len(result))
	print (result)

def main():
	testfile = sys.argv[1]
	outfile  = sys.argv[2]

	x_test_bin, x_test_gau = read(testfile)
	
	Covar  = n.loadtxt('Covariance', delimiter=',')
	mean_0 = n.loadtxt('mean_0', delimiter=',').reshape((1,1))
	mean_1 = n.loadtxt('mean_1', delimiter=',').reshape((1,1))

	P_class0 = n.loadtxt('P_class0', delimiter=',')
	P_class1 = n.loadtxt('P_class1', delimiter=',')
	p_features_bin_0 = n.loadtxt('p_features_bin_0', delimiter=',').reshape((x_test_bin.shape[0],1))
	p_features_bin_1 = n.loadtxt('p_features_bin_1', delimiter=',').reshape((x_test_bin.shape[0],1))

	result = []

	for i in range(x_test_bin.shape[1]):
		prob_0 = P_class0*dot(p_features_bin_0, x_test_bin[:, i]) / (P_class0*dot(p_features_bin_0, x_test_bin[:, i]) + P_class1*dot(p_features_bin_1, x_test_bin[:, i]))
		# print (prob_0);
		if prob_0 > 0.5:
			result.append(0)
		else:
			result.append(1)
	result = n.asarray(result)

	file = open(outfile, "w+")
	file.write('id,label\n')
	for i in range(len(result)):
		file.write(str(i+1)+','+str(result[i])+'\n')

	# print (mean_0)
	# print (mean_1)
	# print (P_class0)
	# print (P_class1)
	# print (p_features_bin_0)
	# print (p_features_bin_1)

if __name__ == "__main__":
	main()
	# test_valid()