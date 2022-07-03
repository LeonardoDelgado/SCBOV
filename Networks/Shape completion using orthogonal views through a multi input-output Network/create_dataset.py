from os import chdir, listdir, getcwd
from os.path import isdir, splitext
import pickle
base_path = getcwd()
name_extension = '.pcd'
samples_x = 'x' + name_extension
labels_y = 'y' + name_extension
pd_file = 'pc' + name_extension
dic_x = {}
dic_y = {}
dic_pd = {}
name_clouds_dir = 'pointclouds'
path_root = '/media/leo/Datos/Respaldo/Old/Descargas/ycb_dataset/ycb'#'/media/leo/Datos/Respaldo/Old/Descargas/grasp_database/grasp_database'#'/media/leo/Datos/Respaldo/Old/Descargas/ycb_dataset/ycb'#'/media/leo/Data/ycb_dataset/ycb'#'/media/leo/Datos1/Respaldo/Old/Descargas/grasp_database/grasp_database'
chdir(path_root)
files_in_root = listdir()
directories = []
for name in files_in_root:
	if isdir(name):
		chdir(name)
		files_inside_name = listdir()
		for file_inside_name in files_inside_name:
			if file_inside_name == name_clouds_dir and isdir(file_inside_name):
				chdir(file_inside_name)
				files_names = listdir()
				for file in files_names:
					path = path_root+'/'+name+'/'+file_inside_name+'/'+file
					if samples_x in file:
						dic_x[name + '/' + file[:-5]] = path          
					elif labels_y in file:
						dic_y[name + '/' + file[:-5]] = path
					elif pd_file in file:
						dic_pd[name + '/' + file[:-6]] = path
				chdir(path_root+'/'+name)
		chdir(path_root+'/')
chdir(base_path)

#directorios con ejemplos de entrenamiento
f = open('/media/leo/Datasets/dic_x_ycb_dataset.pkl','wb')
pickle.dump(dic_x,f)
f.close

#directorios con etiquetas de entrenamiento
f = open('/media/leo/Datasets/dic_y_ycb_dataset.pkl','wb')
pickle.dump(dic_y,f)
f.close


#directorios con ejemplos de pc
f = open('/media/leo/Datasets/dic_pd_ycb_dataset.pkl','wb')
pickle.dump(dic_pd,f)
f.close

