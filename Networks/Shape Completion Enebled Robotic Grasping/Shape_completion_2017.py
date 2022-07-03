import curvox.pc_vox_utils
import pcl
import numpy as np
from reconstruction_network import get_model as load_MFSC
from etxt import getviews_from_voxel_cube
from utilities_for_data import views_to_boxel
from utilities_for_data import boxel_to_cloud

#PC only
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

name = 'donut_poisson_009'
x_filepath = '/media/leo/Datos/Respaldo/Old/Descargas/grasp_database/grasp_database/'+name+'/pointclouds/_0_4_5_x.pcd'
x_np_pts = pcl.load(x_filepath).to_array()
print(x_np_pts.shape)
patch_size = 40
partial_vox = curvox.pc_vox_utils.pc_to_binvox_for_shape_completion(points=x_np_pts, patch_size = patch_size)
batch_x = np.zeros((1, patch_size, patch_size, patch_size, 1),
          dtype=np.float32)
batch_x[0, :, :, :, 0] = partial_vox.data
MFSC = load_MFSC()
MFSC.load_weights('/media/leo/Datasets/best_weights.h5')
print('done')
pred = MFSC.predict(batch_x)
pred_as_b012c = pred.reshape(1, patch_size, patch_size,
	patch_size, 1)
completed_region = pred_as_b012c[0, :, :, :, 0]
print(completed_region.shape)

views = getviews_from_voxel_cube(completed_region,patch_size)


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
n = 6
plt.figure(figsize = (6,6))
for i in range(n):
	ax = plt.subplot(1, n, i+1)
	plt.imshow(views[:,:,i].reshape(patch_size,patch_size))
	
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

plt.show()

voxel_cube = views_to_boxel(views,patch_size)
x_array, y_array, z_array = boxel_to_cloud(voxel_cube)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_array, y_array, z_array, c = 'r', marker = '.')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.get_zaxis().set_visible(False)
	
plt.show()
    