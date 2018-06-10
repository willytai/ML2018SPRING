import sys
import numpy as np
import util as utl

from keras import Model
from keras.layers import Flatten, Embedding, Input, Add, Dot, merge, concatenate, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.utils import plot_model

test_path  = sys.argv[1]
pred_path  = sys.argv[2]
mov_path   = sys.argv[3]
usr_path   = sys.argv[4]
train_path = sys.argv[5]

def CreateModel():
	###########################
	## parameter specification 
	###########################
	drop_rate     = 0.5
	Embedding_dim = 5
	user_num      = 6041
	movie_num     = 3953
	occup_num     = 21
	dense_num     = 2

	# cat      = Input(shape=(18,), name='catogory')
	# cat_em   = Embedding(input_dim=2, output_dim=Embedding_dim)(cat)
	# cat_fl   = Flatten()(cat)
	# cat_den  = Dense(128, activation='relu')(cat)
	# cat_den  = BatchNormalization()(cat_den)
	# cat_den  = Dropout(drop_rate)(cat_den)
	
	# gender     = Input(shape=(1,), name='gender')
	# gender_den = Dense(32, activation='relu')(gender)
	# gender_den = BatchNormalization()(gender_den)
	# gender_den = Dropout(drop_rate-0.2)(gender_den)

	age      = Input(shape=(1,), name='age')
	age_den  = Dense(128, activation='relu')(age)
	# age_den  = BatchNormalization()(age_den)
	# age_den  = Dropout(drop_rate)(age_den)

	occup     = Input(shape=(1,), name='occupation')
	occup_em  = Embedding(input_dim=occup_num, output_dim=Embedding_dim)(occup)
	occup_bi  = Embedding(input_dim=occup_num, output_dim=Embedding_dim)(occup)
	occup_den = Flatten()(occup_em)
	occup_den = Dense(128, activation='relu')(occup_den)
	# occup_den = BatchNormalization()(occup_den)
	# occup_den = Dropout(drop_rate)(occup_den)

	user     = Input(shape=(1,), name='user')
	user_em  = Embedding(input_dim=user_num, output_dim=Embedding_dim)(user)
	user_bi  = Embedding(input_dim=user_num, output_dim=1)(user)

	movie    = Input(shape=(1,), name='movie')
	movie_em = Embedding(input_dim=movie_num, output_dim=Embedding_dim)(movie)
	movie_bi = Embedding(input_dim=movie_num, output_dim=1)(movie)

	dot      = Dot(axes=-1)([user_em, movie_em])
	add      = Add()([dot, user_bi, movie_bi])
	flat     = Flatten()(add)

	concat   = concatenate([age_den, cat_den])
	dense0   = Dense(256, activation='relu')(concat)
	dense0   = BatchNormalization()(dense0)
	dense0   = Dropout(drop_rate)(dense0)

	# concat   = concatenate([dense0, flat])
	concat   = concatenate([cat_den, flat])
	# concat   = flat

	dense    = Dense(256, activation='relu')(concat)
	dense    = BatchNormalization()(dense)
	dense    = Dropout(drop_rate)(dense)

	for i in range(dense_num-1):
		dense = Dense(512, activation='relu')(dense)
		dense = BatchNormalization()(dense)
		dense = Dropout(drop_rate)(dense)

	dense = Dense(1, activation='relu')(dense)

	model = Model([user, movie, age, occup], dense)

	# plot_model(model, to_file='network.png')

	return model


def main():
	#######################
	## training parameters
	#######################
	epochs     = 100
	batch_size = 512
	opt        = 'adam'
	loss       = 'mse'

	usr_dict     = utl.read_dict(usr_path)
	mov_dict     = utl.read_dict(mov_path)
	cat_list     = utl.check_catagory(mov_dict)
	train, label = utl.read_train(train_path)

	usr_train, mov_train = utl.generator(usr_dict, mov_dict, cat_list, train, mode='best')
	# label = utl.norm(label)

	# create model
	model = CreateModel()
	model.summary()

	###############
	## save model
	###############
	model_json = model.to_json()
	with open("model_best.json", "w") as json_file:
		json_file.write(model_json)

	# define callbacks
	filepath   = "weights_best.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
	earlystop  = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
	callbacks_list = [checkpoint, earlystop]

	model.compile(loss=loss, optimizer=opt)
	history = model.fit([usr_train[:, -1], mov_train[:, -1], usr_train[:, :1], usr_train[:, :2]], 
						label, 
						validation_split=0.005, 
						epochs=epochs, 
						batch_size=batch_size,
						callbacks=callbacks_list)


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


if __name__ == '__main__':
	main()