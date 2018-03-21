from sklearn.externals import joblib
import numpy as n
import sys

if len(sys.argv) != 3:
    print ('numbers of arguments does not match!')
    sys.exit()

file_in = sys.argv[1]
file_out = sys.argv[2]

use = [9,8]

raw_test = []
id_x = []
with open (file_in, 'r', encoding='utf-8', errors='ignore') as f:
	for line in f:
		line = line.split(',')
		if len(id_x) == 0 or line[0] != id_x[-1] :
			id_x.append(line[0])
		del line[0]; del line[0]
		raw_test.append(line)

test_in =[]
for i in range(260):
	tmp = []
	for k in use:
		for j in range(9):
			if k == 100:
				tmp.append(float(raw_test[8+i*18][j])**2)
			elif k == 102:
				tmp.append(float(raw_test[8+i*18][j])-float(raw_test[11+i*18][j]))
			elif k == 101:
				tmp.append(float(raw_test[9+i*18][j])**2)
			elif k == 103:
				tmp.append(float(raw_test[8+i*18][j])**2)
			elif k == 104:
				tmp.append(float(raw_test[7+i*18][j])**2)
			elif raw_test[k+18*i][j] == 'NR' or raw_test[k+18*i][j] == 'NR\n':
				tmp.append(0.0)
			else:
				tmp.append(float(raw_test[k+18*i][j]))
	test_in.append(n.asarray(tmp))
test_in = n.asarray(test_in)

# preprocessing
for row in range(test_in.shape[0]):
	for col in range(test_in.shape[1]):
		if test_in[row][col] <= 0:
			count = 0
			while test_in[row][col+count] <= 0:
				count += 1
				if col+count == test_in.shape[1]:
					break
			if col+count == test_in.shape[1]:
				for a in range(count):
					test_in[row][col+a] = test_in[row][col-1]
			else:
				for a in range(count):
					test_in[row][col+a] = (test_in[row][col-1] + test_in[row][col+count]) / 2


model = joblib.load('model.npy')
result = model.predict(test_in)

print (result)

file = open (file_out, "w+")
file.write('id,value\n')
for i in range(len(result)):
	file.write('id_'+str(i)+','+str(result[i][0])+'\n')