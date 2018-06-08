import util as utl
import numpy as np
import sys

from keras.models import model_from_json


test_path  = sys.argv[1]
pred_path  = sys.argv[2]
mov_path   = sys.argv[3]
usr_path   = sys.argv[4]


json_file = open('model_MF.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights('weights_MF.hdf5')

model.summary()

model.compile(loss='mse', optimizer='adam')

test = utl.read_test(test_path)

test_u = test[:, 0]
test_m = test[:, 1]


result = model.predict([test_u, test_m], verbose=1).reshape(-1)

file = open(pred_path, "w+")
file.write('TestDataID,Rating\n')

for i, value in enumerate(result):
	file.write('{},{}\n'.format(i+1, value))
file.close()
