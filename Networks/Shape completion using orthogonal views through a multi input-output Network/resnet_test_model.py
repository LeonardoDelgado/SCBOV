from keras.models import load_model
import numpy as np
from utilities_for_data import split_views

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


datasets_directory = '/media/leo/Datasets/' 
PATH = 'models 2019 08 15/'
NAME = 'model-100-0.002590.h5'
model = load_model(PATH + NAME)
X_val = np.load(datasets_directory + 'X_val.npy')
view_1_val, view_2_val, view_3_val, view_4_val, view_5_val, view_6_val = split_views(X_val)
X_val = np.array([view_1_val, view_2_val, view_3_val, view_4_val, view_5_val, view_6_val])
print(X_val.shape)

decoded_imgs =  model.predict([view_1_val, view_2_val, view_3_val, view_4_val, view_5_val, view_6_val])
decoded_imgs = np.array(decoded_imgs)
print(decoded_imgs.shape)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

n = 6
img = 90
plt.figure(figsize = (12,4))
for i in range(n):
	ax = plt.subplot(2, n, i+1)
	plt.imshow(X_val[i,img,:,:,:].reshape(32,32))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	ax = plt.subplot(2,n, i + n+1)
	plt.imshow(decoded_imgs[i,img,:,:,:].reshape(32,32))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()
