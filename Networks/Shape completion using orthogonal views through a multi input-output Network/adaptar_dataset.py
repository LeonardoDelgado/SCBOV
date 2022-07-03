from tensorpack.dataflow import LMDBSerializer#, LocallyShuffleData
import numpy as np
import etxt_3 as etxt
import os


lmdb_base_path = '/home/leo/Datasets/YCB_source/'
lmdb_name = ['Training.lmdb','Validation.lmdb','Test.lmdb']

TRAINIG_DIRECTORY = '/home/leo/Datasets/YCB 32'
type_file = ['Training','Validation','Test']








for i in range(len(lmdb_name)):
	ds0 = LMDBSerializer.load(lmdb_base_path+lmdb_name[i], shuffle=False)
	#ds0 = LocallyShuffleData(ds0, buffer_size=2000)
	num_of_e = ds0.size()

	try:
		os.mkdir(TRAINIG_DIRECTORY)
	except:
		print('The folder already exits')
	try:
		os.mkdir(TRAINIG_DIRECTORY + '/' + type_file[i])
	except:
		print('The folder already exits')
	try:
		os.mkdir(TRAINIG_DIRECTORY + '/' + type_file[i] +'/X')#'/Validation/X')
	except:
		print('The folder already exits')
	try:	
		os.mkdir(TRAINIG_DIRECTORY + '/' + type_file[i] +'/Y')#'/Validation/Y')
	except:
		print('The folder already exits')

	ds0.reset_state()
	t_t = 0
	size = 32
	for algo in ds0:
	    t_t += 1
	    porcentaje = t_t/num_of_e*100
	    print('\rCreando: ','%.2f' % porcentaje,'%' ,end="\r") 
	    name,X,Y = algo
	    X_Views = etxt.getviews(X,size)
	    Y_Views = etxt.getviews(Y,size)
	    np.save(TRAINIG_DIRECTORY + '/' + type_file[i] + '/X/'+name+'.npy',np.array(X_Views))
	    np.save(TRAINIG_DIRECTORY + '/' + type_file[i] + '/Y/'+name+'.npy',np.array(Y_Views))


