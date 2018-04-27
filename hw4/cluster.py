import numpy as np
import sys, pickle
from skimage import io
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

##############
# parameters #
##############
# epochs_auto = 2
# batch_size  = 64
# rand_seed   = 99

# print ('loading images...', end='')

# image   = np.load(sys.argv[1])
# train_x = image

# print ('done\nPCA...', end='')

#######
# PCA #
#######

# rescale_train = StandardScaler(copy=False).fit(train_x).transform(train_x)
# pca           = PCA(.99, whiten=True, svd_solver='full')

# pca.fit(rescale_train)

# # save the pca result
# pickle_on = open("parameters/pca.pkl", "wb")
# pickle.dump(pca, pickle_on)
# pickle_on.close()

# dim_reduced_train = pca.transform(rescale_train)

# print ('done\nclustering...', end='')

####################
# cluster, k-means #
####################

# kmeans = KMeans(n_clusters=2, n_jobs=-1, random_state=rand_seed).fit(dim_reduced_train)
# pickle_on = open("parameters/kmeans.pkl", "wb")
# pickle.dump(kmeans, pickle_on)
# pickle_on.close()

# print ('done\npredicting...', end='')


# cluster0 = 0
# cluster1 = 0
# for i in range(len(kmeans.labels_)):
# 	if kmeans.labels_[i] == 0:
# 		cluster0 += 1
# 	else:
# 		cluster1 += 1
# print ('cluster0', cluster0)
# print ('cluster1', cluster1)
# sys.exit()



###########
# predict #
###########

same = 0
diff = 0

kmeans = pickle.load(open("parameters/kmeans.pkl", "rb")).labels_
test = np.genfromtxt(sys.argv[2], skip_header=1, delimiter=',').astype(int)
file = open(sys.argv[3], "w+")
file.write("ID,Ans\n")
for i in range(test.shape[0]):
	file.write(str(test[i][0])+',')
	if kmeans[int(test[i][1])] == kmeans[int(test[i][2])]:
		file.write(str(1)+'\n')
		same += 1
	else:
		file.write(str(0)+'\n')
		diff += 1
file.close()

print ('done')

print ('same count', same)
print ('diff count', diff)


###############
# autoencoder #
###############
# encode_dim  = 32
# input_img   = Input(shape=(784, ))
# encode      = Dense(encode_dim, activation='relu')(input_img)
# decode      = Dense(784, activation='sigmoid')(encode)
# autoencoder = Model(input_img, decode)
# encoder     = Model(input_img, encode)
# decode_in   = Input(shape=(encode_dim, ))
# decoder     = Model(decode_in, autoencoder.layers[-1](decode_in))

# autoencoder.compile(optimizer='adam', loss='mse')
# autoencoder.fit(train_x, train_x, epochs=epochs_auto, batch_size=batch_size, validation_data=(test_x, test_x))

# decode_img_test = autoencoder.predict(test_x[0].reshape(-1))
# decode_img_train= autoencoder.predict(train_x[10].reshape(-1))

# io.imsave("train.jpg", train_x[10].reshape((28, 28)))
# io.imsave("test.jpg", test_x[0].reshape((28, 28)))
# io.imsave("train_auto.jpg", decode_img_train.reshape((28, 28)))
# io.imsave("test_auto.jpg", decode_img_test.reshape((28, 28)))