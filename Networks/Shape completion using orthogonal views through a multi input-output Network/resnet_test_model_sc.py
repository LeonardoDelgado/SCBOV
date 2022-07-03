from keras.models import load_model
import numpy as np
from utilities_for_data import split_views
from utilities_for_data import views_to_boxel
from utilities_for_data import boxel_to_cloud
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from DataGenerator import DataGenerator


#PC only
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


datasets_directory = '/media/leo/Datasets/' 
PATH = 'models laptop2 2020 02 16/'
NAME = 'model-174-103.734970.h5'
model = load_model(PATH + NAME)

path_list_IDs = {'train':'/home/leo/Dropbox/CNN for affordances detection/Training_shapenet_32.txt',
             'validation':'/home/leo/Dropbox/CNN for affordances detection/Validation_shapenet_32.txt'}

partition = {'train':'/home/leo/Datasets/Training_32/Training',
             'validation':'/home/leo/Datasets/Training_32/Validation'}

params = {'dim': (32,32),
          'batch_size': 64,
          'n_channels': 6,
          'shuffle': True}

training_generator = DataGenerator(partition['train'], path_list_IDs['train'],**params)
validation_generator = DataGenerator(partition['validation'],path_list_IDs['validation'], **params)



decoded_imgs =  model.predict([viewx_1_val, viewx_2_val, viewx_3_val, viewx_4_val, viewx_5_val, viewx_6_val])
decoded_imgs = np.concatenate(decoded_imgs,axis=3)
print(decoded_imgs.shape)


np.save('/media/leo/Datasets/results_of_proposal.npy',decoded_imgs)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
n = 6
img = 10
plt.figure(figsize = (18,6))
for i in range(n):
	ax = plt.subplot(3, n, i+1)
	plt.imshow(X[img,:,:,i].reshape(32,32))
	
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	ax = plt.subplot(3,n, i + n+1)
	plt.imshow(decoded_imgs[img,:,:,i].reshape(32,32))
	
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	ax = plt.subplot(3,n, i + 2*n+1)
	plt.imshow(Y[img,:,:,i].reshape(32,32))
	
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()

voxel_cube = views_to_boxel(decoded_imgs[img,:,:,:],32)
x_array, y_array, z_array = boxel_to_cloud(voxel_cube)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_array, y_array, z_array, c = 'r', marker = '.')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.get_zaxis().set_visible(False)
	
plt.show()
    
