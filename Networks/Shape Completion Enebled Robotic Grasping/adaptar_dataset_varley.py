from tensorpack.dataflow import LMDBSerializer, LocallyShuffleData
import curvox.pc_vox_utils
import numpy as np
import os



def read_data(x_np_pts,patch_size):
	partial_vox = curvox.pc_vox_utils.pc_to_binvox_for_shape_completion(points=x_np_pts, patch_size = patch_size)
	batch_x = np.zeros((patch_size, patch_size, patch_size, 1),
	      dtype=np.float32)
	batch_x[:, :, :, 0] = partial_vox.data
	return batch_x

lmdb_path = '/home/leo/Datasets/Test_shapenet.lmdb'
ds0 = LMDBSerializer.load(lmdb_path, shuffle=False)
#ds0 = LocallyShuffleData(ds0, buffer_size=2000)
type_VT = 'Test'
num_of_e = ds0.size()


TRAINIG_DIRECTORY = '/home/leo/Datasets/Test_shapenet_varley'
#TEST_DIRECTORY = '/media/leo/Datos/Datasets/'+'Test'

try:
	os.mkdir(TRAINIG_DIRECTORY)
except:
	print('The folder already exits')
try:
	os.mkdir(TRAINIG_DIRECTORY+'/'+type_VT)
except:
	print('The folder already exits')
try:
	os.mkdir(TRAINIG_DIRECTORY+'/'+type_VT+'/X')
except:
	print('The folder already exits')
try:	
	os.mkdir(TRAINIG_DIRECTORY+'/'+type_VT+'/Y')
except:
	print('The folder already exits')

ds0.reset_state()
t_t = 0
size = 40
for algo in ds0:
    t_t += 1
    porcentaje = t_t/num_of_e*100
    print('\rCreando: ','%.2f' % porcentaje,'%' ,end="\r") 
    name,X,Y = algo
    X_Views = read_data(X,size)
    Y_Views = read_data(Y,size)
    np.save(TRAINIG_DIRECTORY+'/'+type_VT+'/X/'+name+'.npy',np.array(X_Views))
    np.save(TRAINIG_DIRECTORY+'/'+type_VT+'/Y/'+name+'.npy',np.array(Y_Views))

