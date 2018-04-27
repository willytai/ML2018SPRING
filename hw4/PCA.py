import numpy as np
from skimage import io
import pickle, sys

# load images
loadpath = str(sys.argv[1])+'/*'
# print (loadpath); sys.exit()
IC = io.imread_collection(loadpath)

#################
# caculate mean #
#################

mean = np.zeros((600, 600, 3))
for image in IC:
	mean += np.asarray(image)
mean   /= 415


##################################################
# construct image vector after substracting mean #
##################################################

img_vec = []
count   = 1
for image in IC:
	print ('processing image', count); count += 1
	img_vec.append((np.asarray(image) - mean).reshape(-1))
img_vec = np.array(img_vec)
img_vec = img_vec.T


###################
# save mean image #
###################

# to_int = lambda t : int(t)
# vfunc  = np.vectorize(to_int)
# mean   = vfunc(mean)
# io.imsave("mean.jpg", mean)


###############
# perform SVD #
###############

eigen_vec, eigen_val, notimportant = np.linalg.svd(img_vec, full_matrices=False)

print ("eignvector", eigen_vec.shape)

# eigen_vec = eigen_vec[:, :100]

# pickle_on = open('eignvector.pkl', "wb")
# pickle.dump(eigen_vec, pickle_on)
# pickle_on.close()


########################
# retrieve eigenvector #
########################
# eigen_vec = pickle.load(open('parameters/eignvector.pkl', "rb"))
# e_face = []
# for i in range(4):
# 	M_10 = -1 * eigen_vec[:, i].reshape(-1)
# 	M_10 -= np.min(M_10)
# 	M_10 /= np.max(M_10)
# 	M_10 = (255*M_10).astype(np.uint8).reshape((600, 600, 3))
# 	e_face.append(M_10)
	# io.imsave("eigenfaces/eigenface_{}.jpg".format(i+1), M_10)


########################
# retrieve eigenvalue  #
########################
# eigen_val = pickle.load(open('parameters/eignvalue.pkl', "rb"))
# eigenface_proportion = []
# sum = eigen_val.sum()
# for i in range(10):
# 	eigenface_proportion.append(eigen_val[i] / sum)

##################
# reconstruction #
##################
savepath = 'reconstruction.jpg'
dimension = 4
img = io.imread(sys.argv[2])
img = img.reshape(-1)
mean= mean.reshape(-1)
u   = []
for dim in range(dimension):
	u.append((img - mean).dot(eigen_vec[:, dim].reshape(-1)))
reconstruct = np.zeros((600*600*3))
for i in range(len(u)):
	reconstruct += u[i]*eigen_vec[:, i]
reconstruct += mean
reconstruct -= np.min(reconstruct)
reconstruct /= np.max(reconstruct)
reconstruct = (255 * reconstruct).astype(np.uint8).reshape((600, 600, 3))
io.imsave(savepath, reconstruct)