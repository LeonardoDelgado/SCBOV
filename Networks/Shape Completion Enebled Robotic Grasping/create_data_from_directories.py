from os import chdir, listdir, getcwd
import etxt_2 as etxt
import pcl

path_root = '/home/leo/Datasets/Training/Training/'
path_to_file_names = '/home/leo/Datasets/Training_list.txt'
Folders = ['X','Y']
f = open(path_to_file_names)
names = f.readlines()
samples = []
labels = []
t_t = 0
num_of_e = len(names)
for name in names:
	t_t += 1
	porcentaje = t_t/num_of_e*100
	print('\rCreando: ','%.2f' % porcentaje,'%' ,end=" ") 
	
	name_int  = int(name)
	x_np_pts = pcl.load(path_root+Folders[0]+'/'+name_int.__str__()+'.pcd').to_array()
	y_np_pts = pcl.load(path_root+Folders[1]+'/'+name_int.__str__()+'.pcd').to_array()
	samples.append(etxt.getviews(x_np_pts,size))
	labels.append(etxt.getviews(y_np_pts,size))

samples = np.array(samples)
labels = np.array(labels)
np.save('/media/leo/Datasets/samples_grasp_database_training.npy',samples)
np.save('/media/leo/Datasets/labels_grasp_database_training.npy',labels)
