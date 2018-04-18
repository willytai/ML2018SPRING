import numpy as n
import sys
import scipy.misc
# from random import randint

def read(filename, filetype):   # returns train_x and train_y y if filetype=='train', otherwise returns test_x
	print ('reading data...', end='')

	train_x = []
	if filetype == 'train':
		train_y = []

	header = True
	with open(filename) as f:
		for line in f:
			if header == True:
				header = False
				continue
			line = line.split(',')
			if filetype == 'train':
				train_y.append(line[0])
			train_x.append(line[1].split(' '))
    
	if filetype == 'train':
		train_y = n.asarray(train_y, dtype='int')
		train_y = train_y.reshape((len(train_y),1))
    
	train_x = n.asarray(train_x, dtype='float32').reshape((len(train_x), 48, 48, 1))

	print ('done')
    
	if filetype == 'train':
		return train_x, train_y
	else:
		return train_x

def array2Img(arr, filename):    # saves arr to a image file
	arr = arr.reshape((48,48))
	scipy.misc.imsave(filename, arr)

def normImg(data, GCN=False):
	print("normalizing data...", end='')
	if GCN == False:
		print ('done')
		return data/225

def main():
	train_x, train_y = read('train.csv', filetype='train')
	print (train_x)
	print ('shape', train_x.shape)
	print (train_y)
	print ('shape', train_y.shape)

if __name__ == '__main__':
	main()
else:
	print ('using functions in utility...')
