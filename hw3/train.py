import utility as utl
import sys
import numpy as n

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
from keras import optimizers
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

def split(x, y, portion):
	datanum  = int(x.shape[0]*(portion))

	valid_x  = x[:datanum, :, :, :]
	valid_y  = y[:datanum, :]
	x        = x[datanum:, :, :, :]
	y        = y[datanum:, :]

	return (x, y), (valid_x, valid_y)

def createModel():
	model = Sequential()

	# convolution network
	model.add(Conv2D(64, kernel_size=(5, 5), padding='same', input_shape=(48,48,1)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.1))
	
	model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.1))

	model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))



	# fully connected network
	model.add(Flatten())
	model.add(Dropout(0.2))

	model.add(Dense(units=512))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add(Dense(units=512))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add(Dense(units=7, activation='softmax'))

	return model

def main():
	trainfilepath = sys.argv[1];
	train_x, train_y = utl.read(trainfilepath, filetype='train')
	(train_x, train_y), (valid_x, valid_y) = split(train_x, train_y, 0.1)

	train_x          = utl.normImg(train_x, GCN=False)
	train_y          = np_utils.to_categorical(train_y, 7)
	valid_x          = utl.normImg(valid_x, GCN=False)
	valid_y          = np_utils.to_categorical(valid_y, 7)

	# create and compile the model
	model = createModel()
	optimizer = optimizers.Adam()
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	# print summary
	model.summary()

	# create check point, set early stop
	filepath   = "my_model.h5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	earlystop  = EarlyStopping(monitor='val_acc', patience=20, mode='max', verbose=1)
	callbacks_list = [earlystop, checkpoint]


	# data augmentation
	epochs = 170
	samples_per_epoch = 10*train_y.shape[0]
	batch_size = 512
	datagen = ImageDataGenerator(
		shear_range=0.2,
		zoom_range=0.2,
		fill_mode='nearest',
    	width_shift_range=0.2,
    	height_shift_range=0.2,
        rotation_range=45,
        horizontal_flip=True)
	datagen.fit(train_x)

	# train
	history = model.fit_generator(datagen.flow(train_x, train_y, batch_size),
								  samples_per_epoch=samples_per_epoch,
								  epochs=epochs,
								  validation_data=(valid_x, valid_y), callbacks=callbacks_list)

	# Evaluate the model
	scores = model.evaluate(valid_x, valid_y)
	print('Loss: %.3f' % scores[0])
	print('Accuracy: %.3f' % scores[1])



	############################Plot Curves############################
	# Loss Curves
	# plt.figure(figsize=[8,6])
	# plt.plot(history.history['loss'],'r',linewidth=3.0)
	# plt.plot(history.history['val_loss'],'b',linewidth=3.0)
	# plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
	# plt.xlabel('Epochs ',fontsize=16)
	# plt.ylabel('Loss',fontsize=16)
	# plt.title('Loss Curves',fontsize=16)
	# plt.savefig('loss.png')
	 
	# Accuracy Curves
	# plt.figure(figsize=[8,6])
	# plt.plot(history.history['acc'],'r',linewidth=3.0)
	# plt.plot(history.history['val_acc'],'b',linewidth=3.0)
	# plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
	# plt.xlabel('Epochs ',fontsize=16)
	# plt.ylabel('Accuracy',fontsize=16)
	# plt.title('Accuracy Curves',fontsize=16)
	# plt.savefig('accuracy.png')

if __name__ == '__main__':
	main()
