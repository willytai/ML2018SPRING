import numpy as np
import sys, re

def norm(data):
	data = data.reshape(-1)
	mean = data.mean()
	std  = data.std()
	return (data - mean ) / std

def read_dict(filename):

	print ('Reading from {}'.format(filename))
	##########################################
	## this function returns a dictionary
	## with ID as key and attribute as values
	##########################################

	data = dict()
	skip = True
	with open(filename, errors='ignore') as f:

		for line in f:

			# skip header
			if skip:
				skip = False
				continue

			line = line.split('::')

			# remove \n
			line[-1] = re.sub(r"\n", "", line[-1])

			# split movie catogory
			tmp = line[-1].split("|")

			attribute  = [line[i] for i in range(1, len(line)-1)]
			attribute += [fuck    for fuck in tmp]
			data[line[0]] = attribute


	return data

def read_train(filename):

	print ('Reading from {}'.format(filename))
	#############################
	## reads the training data
	## collects the trainig pair
	## column 0 for user
	## column 1 for movie
	#############################

	
	data  = np.genfromtxt(filename, delimiter=',', dtype=np.int32, skip_header=True)

	train = data[:, 1:3]
	label = data[:, 3]

	p     = np.random.permutation(len(train))
	train = train[p]
	label = label[p]

	return train, label

def read_test(filename):

	print ('Reading from {}'.format(filename))

	data = np.genfromtxt(filename, delimiter=',', dtype=np.int32, skip_header=True)

	test = data[:, 1:]

	return test

def check_catagory(movie):

	###########################
	## there are 18 catagories
	## for the movies
	###########################

	cat_list = []
	for key, value in movie.items():
		for i in range(1, len(value)):
			if value[i] not in cat_list:
				cat_list.append(value[i])
	return cat_list

def generator(usr_dict, mov_dict, cat_list, train, mode='MF'):

	print ('Generating Data...')
	############################################
	## this function generates 
	## the training data from the training pair
	############################################

	usr_train = []
	mov_train = []

	for pair in train:
		
		usr = pair[0]
		mov = pair[1]

		usr_attribute = usr_dict[str(usr)][:-1] # zip-codes are ignored
		mov_attribute = mov_dict[str(mov)]

		# get the features for users
		# the last one is user id
		usr_features     = np.zeros(4)
		usr_features[-1] = int(usr)
		for i, attr in enumerate(usr_attribute):
			if attr == 'M':
				usr_features[0] = 1
			elif attr != 'F':
				usr_features[i] = int(attr)

		# get the features for movies
		# 18 catogories, movie id
		mov_features     = np.zeros(19)
		mov_features[-1] = int(mov)
		for i, attr in enumerate(mov_attribute):
			try:
				idx = cat_list.index(attr)
				mov_features[idx] = 1
			except ValueError:
				continue

		if mode == 'MF':
			usr_train.append(usr_features[-1])
			mov_train.append(mov_features[-1])
		else:
			usr_train.append(usr_features)
			mov_train.append(mov_features)

	return np.array(usr_train), np.array(mov_train)

def main():
	usr_dict = read_dict('data/users.csv')
	mov_dict = read_dict('data/movies.csv')
	train, _ = read_train('data/train.csv')
	cat_list = check_catagory(mov_dict)
	
	usr_train, mov_train = generator(usr_dict, mov_dict, cat_list, train, mode='best')

	print ('max user id ', usr_train[:, -1].max()) # 6040
	print ('max movie id', mov_train[:, -1].max()) # 3952
	print ('max age     ', usr_train[:, 1].max())  # 56
	print ('max occup id', usr_train[:, 2].max())  # 20

if __name__ == '__main__':
	main()