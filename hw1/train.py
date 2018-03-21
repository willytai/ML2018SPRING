import numpy as n
import sys

def check(i) :
	if i < 471:
		return True
	j = i
	while(j > 480):
		j = j - 480
	if j > 470:
		return False


raw_data = []
with open('../data/train.csv', 'r', encoding='utf-8', errors='ignore') as f :
	for line in f :
		line = line.split(',')
		del line[0]; del line[0]; del line[0]
		raw_data.append(line)
del raw_data[0]

# raw_data[0~17]為第一天的資料
# raw_data[9 + 18 * n] 為PM2.5
# 一天可以提供15筆資料從column 9 ~ 23


# extract the output of the training data
train_out = []
for i in range(240) : # 240 days in total
	for j in range(24) : # 24 datas in one day
		train_out.append(float(raw_data[9+i*18][j]))
train_out = n.asarray(train_out)


# preprocessing
for i in range(len(train_out)):
	if train_out[i] <= 0:
		count = 0
		while train_out[i+count] <= 0:
			count += 1
		for a in range(count):
			train_out[i+a] = (train_out[i-1] + train_out[i+count]) / 2


# feature extraction

use = [9, 8]

x_in = [] # input of the model
for idx in use :
	temp = []
	for i in range(240) :
		for j in range(24) :
			if idx == 100:
				temp.append(float(raw_data[9+i*18][j])**2)
			elif idx == 101:
				temp.append(float(raw_data[8+i*18][j])**2)
			elif raw_data[idx+i*18][j] == 'NR' or raw_data[idx+i*18][j] == 'NR\n':
				temp.append(0.0)
				continue
			else :
				temp.append(float(raw_data[idx+i*18][j]))
	x_in.append(temp)
	x_in[-1] = n.asarray(x_in[-1])
x_in = n.asarray(x_in)

# preprocessing
for row in range(x_in.shape[0]):
	for col in range(x_in.shape[1]):
		if x_in[row][col] <= 0:
			count = 0
			while x_in[row][col+count] <= 0:
				count += 1
			for a in range(count):
				x_in[row][col+a] = (x_in[row][col-1] + x_in[row][i+count]) / 2

#######################################
b = 0.0
reg = 0 # 50

weight_1 = n.ones((len(use), 9))
lr_w_1 = n.ones((len(use), 9))

weight_2 = n.zeros((len(use), 9))
lr_w_2 = n.zeros((len(use), 9))

weight_3 = n.zeros((len(use), 9))
lr_w_3 = n.zeros((len(use), 9))

lr = 1.0
lr_b = 0.0

iterations = int(600)
samples = 5000 # 5760-9 is max

samples_start = 0
samples_end = 5500
# print (len(x_in[0]))
# sys.exit()
#######################################


skip = 0

# TRAIN

for iteration in range(iterations) :
	# initialize gradient
	b_grad = 0.0
	w1_grad = n.zeros((len(use), 9))
	w2_grad = n.zeros((len(use), 9))
	# w3_grad = n.zeros((len(use), 9))

	# start sampling, calculate gradient
	for i in range(samples_start, samples_end) :
		if check(i) == False:
			if iteration == 0:
				skip = skip + 1
			continue;
		dot = 0.0
		for idx in range(len(use)) :
			dot = dot + n.dot(x_in[idx][i:i+9], weight_1[idx])
			if use[idx] == 1010 or use[idx] == 1022:
				dot = dot + n.dot(x_in[idx][i:i+9]**2, weight_2[idx])
				

		b_grad = b_grad - 2*(train_out[i+9] - b - dot)
		for idx in range(len(w1_grad)) :
			w1_grad[idx] = w1_grad[idx] -2*(train_out[i+9] - b - dot)*x_in[idx][i:i+9]
			if use[idx] == 1010 or use[idx] == 1022:
				w2_grad[idx] = w2_grad[idx] -2*(train_out[i+9] - b - dot)*(x_in[idx][i:i+9]**2)
		w1_grad[idx] = w1_grad[idx] + 2*reg*weight_1[idx]
		# w2_grad[idx] = w2_grad[idx] + 2*reg*weight_2[idx]

	# Adagrad
	lr_b = lr_b + b_grad ** 2
	lr_w_1 = lr_w_1 + w1_grad ** 2
	# lr_w_2 = lr_w_2 + w2_grad ** 2

	# update parameter
	b = b - lr/n.sqrt(lr_b) * b_grad
	weight_1 = weight_1 - lr/n.sqrt(lr_w_1) * w1_grad
	for idx in range(len(use)):
		if use[idx] == 1010 or use[idx] == 1022:
			weight_2[idx] = weight_2[idx] - lr/n.sqrt(lr_w_2[idx]) * w2_grad[idx]

	if iteration%10 == 0 :
		loss = 0.0
		for i in range(5010,5510) :
			dot = 0.0
			for idx in range(len(weight_1)) :
				dot = dot + n.dot(x_in[idx][i:i+9], weight_1[idx])
				dot = dot + n.dot(x_in[idx][i:i+9]**2, weight_2[idx])
			loss = loss + (train_out[9+i] - b - dot)**2
		loss = loss / (500)
		print ('loss : ', loss, ' iterantion...', iteration)



# PRINT OUT THE PARAMETERS AND FIVE TEST DATA

print ('b: ', b)
print ('weight_1: ', weight_1)
print ('weight_2: ', weight_2)

print ('test')
dot = 0; test_range = 1000
for i in range(len(weight_1)) :
	dot = dot + n.dot(weight_1[i], x_in[i][test_range:test_range+9])
out = b + dot
dot = 0; test_range = test_range + 1
for i in range(len(weight_1)) :
	dot = dot + n.dot(weight_1[i], x_in[i][test_range:test_range+9])
out1 = b + dot
dot = 0; test_range = test_range + 1
for i in range(len(weight_1)) :
	dot = dot + n.dot(weight_1[i], x_in[i][test_range:test_range+9])
out2 = b + dot
dot = 0; test_range = test_range + 1
for i in range(len(weight_1)) :
	dot = dot + n.dot(weight_1[i], x_in[i][test_range:test_range+9])
out3 = b + dot
dot = 0; test_range = test_range + 1
for i in range(len(weight_1)) :
	dot = dot + n.dot(weight_1[i], x_in[i][test_range:test_range+9])
out4 = b + dot
test_range = test_range - 4
print ('ref   : ', train_out[test_range+9:test_range+14])
print ('result: ', out, ' ', out1, ' ', out2, ' ', out3, ' ', out4)

print('')
print(skip, ' data skipped')

file = open ("bias_v2.txt", "w+")
file.write(str(b))
file.close()

# SAVE THE PARAMETERS
file = open ("weight_1_v2.txt", "w+")
for i in range(len(weight_1)) :
	for idx in range(9) :
		file.write(str(weight_1[i][idx]))
		if idx != 8 :
			file.write(',')
	file.write('\n')

file = open ("weight_2_v2.txt", "w+")
for i in range(len(weight_2)) :
	for idx in range(9) :
		file.write(str(weight_2[i][idx]))
		if idx != 8 :
			file.write(',')
	file.write('\n')
