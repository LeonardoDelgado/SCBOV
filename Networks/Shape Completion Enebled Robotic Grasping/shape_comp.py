# import curvox.pc_vox_utils
# import pcl
import numpy as np
from reconstruction_network import get_model as load_MFSC
# from etxt import getviews_from_voxel_cube
# from utilities_for_data import views_to_boxel
# from utilities_for_data import boxel_to_cloud

#PC only
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

patch_size = 40
datasets_directory = '/media/leo/Datasets/'
X = np.load(datasets_directory + 'samples_40.npy')
print(X.shape)

ejemplos = X.shape[0]
output = []


MFSC = load_MFSC()
MFSC.load_weights('/media/leo/Datasets/best_weights.h5')


for ejemplo in range(ejemplos):
	batch_x = np.zeros((1, patch_size, patch_size, patch_size, 1),
	        dtype=np.float32)
	batch_x[0,:,:,:,0] = X[ejemplo, :,:,:].reshape(40,40,40)
	pred = MFSC.predict(batch_x)
	pred_as_b012c = pred.reshape(1, patch_size, patch_size,
		patch_size, 1)
	completed_region = pred_as_b012c[0, :, :, :, 0]
	output.append(completed_region)
	print(ejemplo)
output = np.array(output)
np.save('output '+'40'+'.npy',output)

# pred_as_b012c = pred.reshape(1, patch_size, patch_size,
# 	patch_size, 1)
# completed_region = pred_as_b012c[0, :, :, :, 0]
# print(completed_region.shape)

# views = getviews_from_voxel_cube(completed_region,patch_size)


# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('tkagg')
# n = 6
# plt.figure(figsize = (6,6))
# for i in range(n):
# 	ax = plt.subplot(1, n, i+1)
# 	plt.imshow(views[:,:,i].reshape(patch_size,patch_size))
	
# 	ax.get_xaxis().set_visible(False)
# 	ax.get_yaxis().set_visible(False)

# plt.show()

# voxel_cube = views_to_boxel(views,patch_size)
# x_array, y_array, z_array = boxel_to_cloud(voxel_cube)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x_array, y_array, z_array, c = 'r', marker = '.')
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# ax.get_zaxis().set_visible(False)
	
# plt.show()
#     