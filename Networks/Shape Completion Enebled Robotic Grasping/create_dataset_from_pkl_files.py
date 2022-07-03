import curvox.pc_vox_utils
import pcl
import numpy as np
from etxt import getviews_from_voxel_cube
from utilities_for_data import views_to_boxel
from utilities_for_data import boxel_to_cloud
import pickle
import numpy as np
import csv

def read_data(x_filepath,patch_size):
	x_np_pts = pcl.load(x_filepath).to_array()
	#print(x_np_pts.shape)
	patch_size = 40
	partial_vox = curvox.pc_vox_utils.pc_to_binvox_for_shape_completion(points=x_np_pts, patch_size = patch_size)
	batch_x = np.zeros((patch_size, patch_size, patch_size, 1),
	      dtype=np.float32)
	batch_x[:, :, :, 0] = partial_vox.data
	return batch_x





def load_data(pd = False):
	with open('/home/leo/Datasets/dic_x_ycb_master_data.pkl', 'rb') as f:
		dic_x = pickle.load(f)
		print('entre')
	with open('/home/leo/Datasets/dic_y_ycb_master_data.pkl', 'rb') as f:
		dic_y = pickle.load(f)
		print('entre')
	if pd == True:
		with open('/home/leo/Datasets/dic_pd_ycb_master_data.pkl', 'rb') as f:
			dic_pd = pickle.load(f)
		keys_pd = dic_pd.keys()
		keys_x = dic_x.keys()
		keys_y = dic_y.keys()
		return keys_x, keys_y, keys_pd, dic_x, dic_y, dic_pd
	keys_x = dic_x.keys()
	keys_y = dic_y.keys()
	return keys_x, keys_y, dic_x, dic_y
	#return keys_y, dic_y
	


if __name__ == '__main__':
	import pcl
	from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
	import etxt
	views = 6
	patch_size = 40
	number = 40
	keys_x, keys_y, dic_x, dic_y = load_data()
	#keys_y, dic_y = load_data()
	samples = []
	labels = []
	cont = 0
	if(keys_x==keys_y):
		keys = list(keys_y)
		num_of_e = len(keys_y)
		t_t = 0
		for key in keys:
			t_t += 1
			porcentaje = t_t/num_of_e*100
			print('\rCreando: ','%.2f' % porcentaje,'%' ,end="\r") 
			y_np_pts = pcl.load(dic_y[key]).to_array()
			labels.append(read_data(dic_y[key],patch_size))
			samples.append(read_data(dic_x[key],patch_size))
	labels = np.array(labels)
	samples = np.array(samples)

	np.save('/home/leo/Datasets/samples_for_traing_varleys_networks'+number.__str__()+'.npy',samples)
	np.save('/home/leo/Datasets/labels_for_traing_varleys_networks'+number.__str__()+'.npy',labels)