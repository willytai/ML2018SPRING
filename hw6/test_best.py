import util as utl
import numpy as np
import sys

from keras.models import model_from_json


test_path  = sys.argv[1]
pred_path  = sys.argv[2]
mov_path   = sys.argv[3]
usr_path   = sys.argv[4]


json_file = open('model_best.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights('weights_best.hdf5')

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

usr_dict = utl.read_dict(usr_path)
mov_dict = utl.read_dict(mov_path)
cat_list = utl.check_catagory(mov_dict)
test     = utl.read_test(test_path)

test_u, test_m = utl.generator(usr_dict, mov_dict, cat_list, test, mode='best')

result = model.predict([test_u[:, -1], test_m[:, -1], test_u[:, 1], test_u[:, 2]], verbose=1).reshape(-1)

# result = result*2
# result += 3

file = open(pred_path, "w+")
file.write('TestDataID,Rating\n')

for i, value in enumerate(result):
	file.write('{},{}\n'.format(i+1, value))
file.close()
