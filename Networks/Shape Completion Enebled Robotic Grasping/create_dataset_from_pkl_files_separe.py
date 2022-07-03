import curvox.pc_vox_utils
import pcl
import numpy as np
from etxt import getviews_from_voxel_cube
from utilities_for_data import views_to_boxel
from utilities_for_data import boxel_to_cloud
import pickle
import numpy as np
import csv
import random
import os
import shutil

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
		random.seed(10)

		random.shuffle(keys)
			############################
		############ Set Traing test
		Pocentaje_training = 0.9
		Porcentaje_validation = 0.1 # Con respecto a training 
		Limite_training = round((num_of_e*Pocentaje_training)*(1 - Porcentaje_validation))
		Limite_validacion = round((num_of_e*Pocentaje_training))

		TRAINIG_DIRECTORY = '/home/leo/Datasets/'+'Training'
		TEST_DIRECTORY = '/home/leo/Datasets/'+'Test'

		try:
			os.mkdir(TRAINIG_DIRECTORY)
		except:
			print('The folder already exits')
		try:	
			os.mkdir(TEST_DIRECTORY)
		except:
			print('The folder already exits')


		try:
			os.mkdir(TRAINIG_DIRECTORY+'/Training')
		except:
			print('The folder already exits')
		try:
			os.mkdir(TRAINIG_DIRECTORY+'/Training/X')
		except:
			print('The folder already exits')
		try:	
			os.mkdir(TRAINIG_DIRECTORY+'/Training/Y')
		except:
			print('The folder already exits')
		try:	
			os.mkdir(TRAINIG_DIRECTORY+'/Validation')
		except:
			print('The folder already exits')
		try:	
			os.mkdir(TRAINIG_DIRECTORY+'/Validation/X')
		except:
			print('The folder already exits')
		try:	
			os.mkdir(TRAINIG_DIRECTORY+'/Validation/Y')
		except:
			print('The folder already exits')
		try:	
			os.mkdir(TEST_DIRECTORY+'/X')
		except:
			print('The folder already exits')
		try:	
			os.mkdir(TEST_DIRECTORY+'/Y')
		except:
			print('The folder already exits')



		for key in keys:
			t_t += 1
			porcentaje = t_t/num_of_e*100
			print('\rCreando: ','%.2f' % porcentaje,'%' ,end="\r") 

			if t_t<Limite_training:
				shutil.copyfile(dic_x[key],TRAINIG_DIRECTORY+'/Training/X/'+t_t.__str__()+'.pcd')
				shutil.copyfile(dic_y[key],TRAINIG_DIRECTORY+'/Training/Y/'+t_t.__str__()+'.pcd')
			elif t_t<Limite_validacion:
				shutil.copyfile(dic_x[key],TRAINIG_DIRECTORY+'/Validation/X/'+t_t.__str__()+'.pcd')
				shutil.copyfile(dic_y[key],TRAINIG_DIRECTORY+'/Validation/Y/'+t_t.__str__()+'.pcd')
			else: 
				shutil.copyfile(dic_x[key],TEST_DIRECTORY+'/X/'+t_t.__str__()+'.pcd')
				shutil.copyfile(dic_y[key],TEST_DIRECTORY+'/Y/'+t_t.__str__()+'.pcd')
			