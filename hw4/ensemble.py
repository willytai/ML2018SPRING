from keras.models import load_model, Model
from keras.layers import *
from keras.backend import argmax
from keras import layers
import sys

filenames = []; models = []
for i in range(1, len(sys.argv)):
	filenames.append(sys.argv[i])

for file in filenames:
	print ('loading ' + file + '...', end='')
	models.append(load_model(file))
	print ('done')

inputs  = Input(shape=models[0].input_shape[1:])

outputs = [model(inputs) for model in models]

yAvg    = layers.average(outputs)

modelEns = Model(inputs=inputs, outputs=yAvg, name='ensemble') 

modelEns.save('merge.h5')