#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:38:43 2020

@author: leo
"""
#from utilities_for_data import views_to_boxel
from utilities_for_data1 import boxel_to_cloud
from utilities_for_data1 import truncate_view

import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import os
import sys
sys.path.append("/home/leo/Documents/pcn")
from tf_util import chamfer
from etxt_3 import getviews_from_voxel_cube, pcdtovoxel, getviews
from io_util import read_pcd


def scale(o,t):
    # print(t.shape)
    # t_max = np.max(t)
    # t_min = np.min(t)
    # t_min_abs = np.abs(t_min)
    # maxi = np.max([t_max,t_min_abs])
    # o_max = np.max(o)
    # o_maxd = o_max/2.0
    # o = ((o-o_maxd)/o_maxd)*maxi
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

## Directory of Cd distance #######################

###########################################
patch_size = 40



MODEL = 'ShapeNet'
SAVE_PATH = '/home/leo/Results/'
BASE_PATH = '/home/leo/Datasets/Test_Varley_' 
    
    
path_list_IDs = BASE_PATH + MODEL + '/' + 'Test.txt'
partition = BASE_PATH + MODEL + '/Test'

RESULTS_PATH = SAVE_PATH + 'Results_Varley_'+MODEL+'.npy'

decoded_imgs = np.load(RESULTS_PATH,allow_pickle=True)
f = open(path_list_IDs)
list_IDs = f.read().splitlines()
f.close()



Target_names = []
for IDS in list_IDs:
    Target_names.append(IDS.replace('/', '_'))

Results = []
Results_MSE = []

for i in range(decoded_imgs.shape[0]):
    sess = tf.Session(config=config)
    print(i)
    name = Target_names[i]
    target = np.load(partition + '/Y/'+ name + '.npy', allow_pickle=True)
    voxel_cube = target[:,:,:,0]
    x_array_o, y_array_o, z_array_o = boxel_to_cloud(voxel_cube)

    decoded_imgsT=getviews(x_np_pts = np.array((x_array_o, y_array_o, z_array_o)).T,scale=32,offset = True)

    target = np.array([x_array_o, y_array_o, z_array_o]).T
    ta = np.zeros((1,*target.shape))
    target_range = read_pcd(os.path.join('/home/leo/Datasets/Test_PCN_'+MODEL+'/Test', 'complete', '%s.pcd' % name.replace('_', '/')))
    target = scale(target,target_range)
    
    
    ta[0,] = target
    
    pred = decoded_imgs[i,:]
    pred_as_b012c = pred.reshape(1, patch_size, patch_size,	patch_size, 1)
    voxel_cube = pred_as_b012c[0, :, :, :, 0]
    x_array_T, y_array_T, z_array_T = boxel_to_cloud(voxel_cube)

    decoded_imgsO=getviews(x_np_pts = np.array((x_array_T, y_array_T, z_array_T)).T,scale=32,offset = True)

    output = np.array([x_array_T, y_array_T, z_array_T]).T
    ot = np.zeros((1,*output.shape))
    output = scale(output,target_range)
    ot[0,] = output

    R = sess.run(chamfer(ot,ta))
	#print(R)
    Results.append(R)
    RMSE = mseIm(truncate_view(decoded_imgsO,32),truncate_view(decoded_imgsT,32))
    print(RMSE,R)
    Results_MSE.append(RMSE)

    tf.reset_default_graph()


np.save(SAVE_PATH + 'Results_CD_Varley_' + MODEL  +'.npy',np.array(Results))
np.save(SAVE_PATH + 'Results_MSE_Varley_' + MODEL  +'.npy',np.array(Results_MSE))
