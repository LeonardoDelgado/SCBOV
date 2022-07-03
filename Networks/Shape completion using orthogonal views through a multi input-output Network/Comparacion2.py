from utilities_for_data1 import views_to_boxel
from utilities_for_data1 import boxel_to_cloud
from utilities_for_data1 import dumi_voxel_cube
from utilities_for_data1 import truncate_view
import os
from etxt_3 import getviews_from_voxel_cube

import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True



import sys
sys.path.append("/home/leo/Documents/pcn")
from tf_util import chamfer
from io_util import read_pcd
## Directory of Cd distance #######################



###########################################3
def scale(o,t):
    # print(t.shape)
    # t_max = np.max(t)
    # t_min = np.min(t)
    # t_min_abs = np.abs(t_min)
    # maxi = np.max([t_max,t_min_abs])
    # o_max = np.max(o)
    # o_maxd = o_max/2.0
    # o = ((o-o_maxd)/o_maxd)#*maxi
    o_max = np.max(o)
    o_min = np.min(o)
    if o_min<=0:
        o = o - o_min
        o = o/(o_max - o_min)
    else:
        o = o - o_min
        o = o/(o_max - o_min)
    return o

def mseIm(i1,i2):
    results = []
    for i in range(i1.shape[2]):
        R = np.mean((i1[:,:,i]/32.0-i2[:,:,i]/32.0)**2)
        print(R)
        results.append(R)
    return np.sum(results)/len(results)


MODEL = 'YCB'
SAVE_PATH = '/home/leo/Results/'
BASE_PATH = '/home/leo/Datasets/Test_Proposal_' 
    
    
path_list_IDs = BASE_PATH + MODEL + '/' + 'Test.txt'
partition = BASE_PATH + MODEL + '/Test'

RESULTS_PATH = SAVE_PATH + 'Results_Proposal_'+MODEL+'.npy'

decoded_imgs = np.load(RESULTS_PATH,allow_pickle=True)

f = open(path_list_IDs)
list_IDs = f.read().splitlines()
f.close()



Target_names = []
for IDS in list_IDs:
    Target_names.append(IDS.replace('/', '/'))








Results = []
Results_MSE = []
for i in range(decoded_imgs.shape[0]):
   sess = tf.Session(config=config)
   print(i)
   name = Target_names[i]
   voxel_cube = dumi_voxel_cube(decoded_imgs[i,:,:,:],32)
   decoded_imgs1=getviews_from_voxel_cube(voxel_cube,32)
   x_array_o, y_array_o, z_array_o = boxel_to_cloud(voxel_cube)
   output = np.array([x_array_o, y_array_o, z_array_o]).T
   target = read_pcd(os.path.join('/home/leo/Datasets/Test_PCN_'+MODEL+'/Test', 'complete', '%s.pcd' % name.replace('_', '/')))
   output = scale(output,target)
   
   target_10 = np.load(partition + '/Y/'+ name.replace('/', '_') + '.npy', allow_pickle=True)
   voxel_cube = views_to_boxel(target_10,32)
   x_array_T, y_array_T, z_array_T = boxel_to_cloud(voxel_cube)
   target_1 = np.array([x_array_T, y_array_T, z_array_T]).T
   print('original',np.min(target),np.max(target))
   target = scale(target_1,target)
   

   
   t = np.zeros((1,*target.shape))
   t[0,]=target
   print('target',np.min(t),np.max(t))
   ot = np.zeros((1,*output.shape))
   print('salida',np.min(t),np.max(t))
   ot[0,] = output

   R = sess.run(chamfer(ot,t))
   RMSE = mseIm(truncate_view(decoded_imgs1,32),truncate_view(target_10,32))
   print('resultado',R)
   print(R)
   Results.append(R)
   Results_MSE.append(RMSE)
   tf.reset_default_graph()
np.save(SAVE_PATH + 'Results_CD_Proposal_Dumi_' + MODEL  +'.npy',np.array(Results))
np.save(SAVE_PATH + 'Results_MSE_Proposal_Dumi_' + MODEL  +'.npy',np.array(Results_MSE))



#
#
#
#

#
Results = []
Results_MSE = []
for i in range(decoded_imgs.shape[0]):
   sess = tf.Session(config=config)
   print(i)
   name = Target_names[i]
   voxel_cube = views_to_boxel(decoded_imgs[i,:,:,:],32)
   x_array_o, y_array_o, z_array_o = boxel_to_cloud(voxel_cube)
   output = np.array([x_array_o, y_array_o, z_array_o]).T
   target = read_pcd(os.path.join('/home/leo/Datasets/Test_PCN_'+MODEL+'/Test', 'complete', '%s.pcd' % name.replace('_', '/')))
   output = scale(output,target)
   
   target_10 = np.load(partition + '/Y/'+ name.replace('/', '_') + '.npy', allow_pickle=True)
   voxel_cube = views_to_boxel(target_10,32)
   x_array_T, y_array_T, z_array_T = boxel_to_cloud(voxel_cube)
   target_1 = np.array([x_array_T, y_array_T, z_array_T]).T
   print('original',np.min(target),np.max(target))
   target = scale(target_1,target)
   

   
   t = np.zeros((1,*target.shape))
   t[0,]=target
   print('target',np.min(t),np.max(t))
   ot = np.zeros((1,*output.shape))
   print('salida',np.min(t),np.max(t))
   ot[0,] = output

   R = sess.run(chamfer(ot,t))
   RMSE = mseIm(truncate_view(decoded_imgs[i,:,:,:],32),truncate_view(target_10,32))
   print('resultado',R)
   print(R)
   Results.append(R)
   Results_MSE.append(RMSE)
   tf.reset_default_graph()
np.save(SAVE_PATH + 'Results_CD_Proposal_' + MODEL  +'.npy',np.array(Results))
np.save(SAVE_PATH + 'Results_MSE_Proposal_' + MODEL  +'.npy',np.array(Results_MSE))
#
#
#
#
#




import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
n = 6

Results = []
for j in range(decoded_imgs.shape[0]):
    print(j)
    name = Target_names[j]
    target = np.load(partition + '/Y/'+ name.replace('/', '_') + '.npy', allow_pickle=True)
   # norm = np.max(target)
   # target = target/norm
    print('target',np.min(target),np.max(target))

    images = decoded_imgs[j,:,:,:]
#    images = images/norm
    print('images',np.min(images),np.max(images))
    fig0 = plt.figure(0,figsize = (18,6))
    for i in range(n):
    	ax = plt.subplot(3, n, i+1)
    	plt.imshow(target[:,:,i])
    	
    	ax.get_xaxis().set_visible(False)
    	ax.get_yaxis().set_visible(False)
    
    	ax = plt.subplot(3,n, i + n+1)
    	plt.imshow(decoded_imgs[j,:,:,i].reshape(32,32))
    	
    	ax.get_xaxis().set_visible(False)
    	ax.get_yaxis().set_visible(False)
    
    	ax = plt.subplot(3,n, i + 2*n+1)
    	plt.imshow(target[:,:,i])
    	
    	ax.get_xaxis().set_visible(False)
    	ax.get_yaxis().set_visible(False)
        #plt.show()
    fig0.show()
    input()
    
    # R = mseIm(images,target)
    # print('resultado',R)
    # #print(R)
    # Results.append(R)

# np.save('/home/leo/Datasets/RasultadosShapeNetMSEProposalF_scale_Dumi.npy',np.array(Results))











