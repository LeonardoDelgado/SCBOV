import os

path_list_IDs = "/home/leo/Dropbox/CNN for affordances detection/Validation_shapenet_32.txt"
f = open(path_list_IDs)
list_IDs = f.read().splitlines()
f.close()
path_root = '/media/leo/Datos/Datasets/Training_32/Validation'
for i, name in enumerate(list_IDs):
    print('con nombre: ',path_root+'/X/'+name+".npy")

    try:
        os.isfile(path_root+'/X/'+name+".npy")
    except:
        print('The file do not exits')
    try:
        os.isfile(path_root+'/Y/'+name+".npy")
    except:
        print('The file do not exits')



