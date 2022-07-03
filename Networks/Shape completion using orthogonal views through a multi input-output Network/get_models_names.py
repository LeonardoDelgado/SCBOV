from os import chdir, listdir, getcwd
from os.path import isdir, splitext
import pickle
import numpy as np
import csv
base_path = getcwd()
name_extension = '.pcd'
samples_x = 'x' + name_extension
labels_y = 'y' + name_extension
pd_file = 'pc' + name_extension
dic_x = {}
dic_y = {}
dic_pd = {}
name_clouds_dir = 'pointclouds'
path_root = '/media/leo/Datos/Respaldo/Old/Descargas/ycb_dataset/ycb'#'/media/leo/Datos1/Respaldo/Old/Descargas/grasp_database/grasp_database'
chdir(path_root)
files_in_root = listdir()
clases = {}
for name in files_in_root:
	if isdir(name):
		if name[0:-12] in clases:
			partial = clases[name]#[0:-12]]
		else: 
			clases[name] = 0#[0:-12]] = 0
			partial = 0
		chdir(name)
		files_inside_name = listdir()
		for file_inside_name in files_inside_name:
			if file_inside_name == name_clouds_dir and isdir(file_inside_name):
				cont = 0
				chdir(file_inside_name)
				files_names = listdir()
				for file in files_names:
					#path = path_root+'/'+name+'/'+file_inside_name+'/'+file
					if samples_x in file:
						cont = cont + 1 
				chdir(path_root+'/'+name)
				clases[name] = partial + cont#[0:-12]] = partial + cont
		chdir(path_root+'/')
chdir(base_path)


csv_file = "Elementos_por_clase_ycb.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, clases.keys())
        writer.writeheader()
        writer.writerow(clases)
except IOError:
    print("I/O error") 







