from os import chdir, listdir, getcwd

filename = "/home/leo/Datasets/YCB 32/Validation.txt"
myfile = open(filename, 'w')
base_path = getcwd()

path_root = '/home/leo/Datasets/YCB 32/Validation/Y'
chdir(path_root)
files_in_root = listdir()


for name in files_in_root:
	myfile.write(name[0:-4]+'\n')
	print(name[0:-4])
chdir(base_path)
myfile.close()


