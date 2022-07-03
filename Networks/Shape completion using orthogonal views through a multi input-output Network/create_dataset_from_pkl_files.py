import pickle
import numpy as np
import csv
def load_data(pd = False):
	with open('/media/leo/Datasets/dic_x_ycb_dataset.pkl', 'rb') as f:
		dic_x = pickle.load(f)
	with open('/media/leo/Datasets/dic_y_ycb_dataset.pkl', 'rb') as f:
		dic_y = pickle.load(f)
	if pd == True:
		with open('/media/leo/Datasets/dic_pd_ycb_dataset.pkl', 'rb') as f:
			dic_pd = pickle.load(f)
		keys_pd = dic_pd.keys()
		keys_x = dic_x.keys()
		keys_y = dic_y.keys()
		return keys_x, keys_y, keys_pd, dic_x, dic_y, dic_pd
	keys_x = dic_x.keys()
	keys_y = dic_y.keys()
	return keys_x, keys_y, dic_x, dic_y
	#return keys_y, dic_y
	
# def ycb(seed = 4,number = 14):
# 	import random
# 	names = []
# 	random.seed(seed)
# 	# with open('Elementos_por_clase_ycb.csv') as File:
# 	# 	reader = csv.reader(File, delimiter=',', quotechar=',',quoting=csv.QUOTE_MINIMAL)
# 	# 	for row in reader:
# 	# 		names.append(row)

# 	names =np.array(names)
# 	names = list(names[0,:])
# 	random.shuffle(names)
# 	names = names[0:number]
# 	return names 

if __name__ == '__main__':
	import pcl
	from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
	import etxt_3 as etxt
	views = 6
	size = 64
	number = 14
	keys_x, keys_y, dic_x, dic_y = load_data()
	# names = ycb(number = number)
	#keys_y, dic_y = load_data()
	samples = []
	labels = []
	keysdic = []
	cont = 0
	if(keys_x==keys_y):
		keys = list(keys_y)
		num_of_e = len(keys)
		t_t=0
		for key in keys:
			t_t += 1
			porcentaje = t_t/num_of_e*100
			print('\rCreando: ','%.2f' % porcentaje,'%' ,end=" ") 
			# print(key)
			# a,b = key.split('/')
			# if a in  names:
			x_np_pts = pcl.load(dic_x[key]).to_array()
			y_np_pts = pcl.load(dic_y[key]).to_array()
			keysdic.append(key)
			samples.append(etxt.getviews(x_np_pts,size))
			labels.append(etxt.getviews(y_np_pts,size))
	#samples = np.array(samples)
	keysdic = np.array(keysdic)
	labels = np.array(labels)

	np.save('/media/leo/Datasets/samples_grasp_database.npy',samples)
	np.save('/media/leo/Datasets/labels_grasp_database.npy',labels)
	#np.save('/media/leo/Datasets/labels_keysdic_grasp_database.npy',keysdic)

	# Visualizar datos como imagenes
	#import matplotlib.pyplot as plt
	#n = 6
	#cons = 7
	#plt.figure(figsize = (6,4))
	#for i in range(n):
	#	ax = plt.subplot(2,n,i+1)
	#	plt.imshow(samples[cons,:,:,i])
	#	plt.gray()
	#	ax.get_xaxis().set_visible(False)
	#	ax.get_yaxis().set_visible(False)

	#	ax = plt.subplot(2, n, i + 1 + n)
	#	plt.imshow(labels[cons,:,:,i])
	#	plt.gray
	#	ax.get_xaxis().set_visible(False)
	#	ax.get_yaxis().set_visible(False)
	#plt.show()


