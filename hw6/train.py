import sys
import numpy as np
import util as utl

from keras import Model
from keras.layers import Flatten, Embedding, Input, Add, Dot, merge
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

train_path = sys.argv[1]

def CreateModel():
	###########################
	## parameter specification 
	###########################
	Embedding_dim = 5
	user_num      = 6041
	movie_num     = 3953

	user     = Input(shape=(1,), name='user')
	user_em  = Embedding(input_dim=user_num, output_dim=Embedding_dim)(user)
	user_bi  = Embedding(input_dim=user_num, output_dim=1)(user)

	movie    = Input(shape=(1,), name='movie')
	movie_em = Embedding(input_dim=movie_num, output_dim=Embedding_dim)(movie)
	movie_bi = Embedding(input_dim=movie_num, output_dim=1)(movie)

	flat_usr = Flatten()(user_em)
	flat_mov = Flatten()(movie_em)

	flat_u_b = Flatten()(user_bi)
	flat_m_b = Flatten()(movie_bi)

	DOT      = Dot(axes=-1)([flat_usr, flat_mov])
	Bias     = Add()([DOT,  flat_u_b, flat_m_b])

	model    = Model([user, movie], Bias)


	return model


def main():
	#######################
	## training parameters
	#######################
	epochs     = 50
	batch_size = 256
	opt        = Adam(8e-4)
	loss       = 'mse'

	train, label = utl.read_train(train_path)

	usr_train = train[:, 0]
	mov_train = train[:, 1]

	# create model
	model = CreateModel()
	model.summary()

	###############
	## save model
	###############
	model_json = model.to_json()
	with open("model_MF.json", "w") as json_file:
		json_file.write(model_json)

	# define callbacks
	filepath   = "weights_MF.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
	earlystop  = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)
	callbacks_list = [checkpoint, earlystop]

	model.compile(loss=loss, optimizer=opt)

	history = model.fit([usr_train, mov_train], 
						label, validation_split=0.1, 
						epochs=epochs, 
						callbacks=callbacks_list, 
						batch_size=batch_size)


	############################Plot Curves############################
	# # Loss Curves
	# plt.figure(figsize=[8,6])
	# plt.plot(history.history['loss'],'r',linewidth=3.0)
	# plt.plot(history.history['val_loss'],'b',linewidth=3.0)
	# plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
	# plt.xlabel('Epochs ',fontsize=16)
	# plt.ylabel('Loss',fontsize=16)
	# plt.title('Loss Curves',fontsize=16)
	# plt.savefig('loss.png')


if __name__ == '__main__':
	main()