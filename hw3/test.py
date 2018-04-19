import numpy as n
import utility as utl
import sys
from keras.models import load_model

def transform(result):
	tmp = []
	for i in range(result.shape[0]):
		max = result[i][0]
		cls = 0
		for classes in range(1,7):
			if max < result[i][classes]:
				max = result[i][classes]
				cls = classes
		tmp.append(cls)
	return n.asarray(tmp)

def save(filename, result):
	file = open(filename, "w+")
	file.write('id,label\n')
	for i in range(len(result)):
		file.write(str(i)+','+str(result[i])+'\n')

def main():
	testfile = sys.argv[1] # the file path of test.csv
	loadmode = sys.argv[2] # the mode to determine which model to load
	filepath = sys.argv[3] # the path to save the prediction file
	test_x 	 = utl.read(testfile, filetype='test')
	test_x   = utl.normImg(test_x, GCN=False)

	if loadmode == 'public':
		model = load_model('merge_tmp.h5')
	else:
		model = load_model('merge.h5')
	result    = model.predict(test_x)
	output    = result.argmax(1)
	save(filepath, output)

if __name__ == '__main__':
	main()