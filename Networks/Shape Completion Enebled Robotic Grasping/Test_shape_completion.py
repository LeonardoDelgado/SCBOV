#import curvox.pc_vox_utils
#import pcl
import numpy as np
#from keras.models import load_model
from reconstruction_network import get_model 
from etxt_3 import getviews_from_voxel_cube
from utilities_for_data1 import views_to_boxel
from utilities_for_data1 import boxel_to_cloud
from DataGenerator_Varley import DataGenerator
import ctypes
ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
#PC only
# import tensorflow as tf
import tensorflow.compat.v1 as tf
with tf.device('/gpu:0'):
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #session = tf.Session(config=config)
    session = tf.Session()

    MODEL = 'YCB'
    SAVE_PATH = '/home/leo/Results/'
    BASE_PATH = '/home/leo/Datasets/Test_Varley_'
    

    PATH = '/home/leo/Models/'
    NAME = 'Model_Varley_'+ MODEL + '.h5'
    model = get_model()
    model.load_weights(PATH + NAME)
    
    
    path_list_IDs = BASE_PATH + MODEL + '/' + 'Test.txt'
    partition = BASE_PATH + MODEL + '/Test'
    params = {'dim': (40,40,40),
              'batch_size': 64,
              'n_channels': 1,
              'shuffle': False}
    
    test_generator = DataGenerator(partition, path_list_IDs,**params)
    decoded_imgs = model.predict_generator(generator=test_generator,
                        use_multiprocessing=True,
                        workers=16)
    decoded_imgs = np.uint8(decoded_imgs>0.49)
    
    print(decoded_imgs.shape)
    np.save(SAVE_PATH + 'Results_Varley_'+MODEL+'.npy',decoded_imgs)
    
    
    
    
    f = open(path_list_IDs)
    list_IDs = f.read().splitlines()
    f.close()
    #
    patch_size = 40
#    import matplotlib.pyplot as plt
#    import matplotlib
#    matplotlib.use('tkagg')
#    
#    # img = 0
#    # for i, id in enumerate(list_IDs):
#    #     if '02691156_1169d987dbbce76775f4ea0b85a53249' == list_IDs[i].replace('/', '_'):
#    #         img = i
#    img = 1
#
#    
#    n=6
#    print(img)
#    name = list_IDs[img].replace('/', '_')
#    print(name)
#    X = np.load(partition + '/X/'+ name + '.npy')
#    X = X[:,:,:,0]
#    X = getviews_from_voxel_cube(X,patch_size)
#    print(X.shape)
#    
#    
#    pred = decoded_imgs[img,:]
#    pred_as_b012c = pred.reshape(1, patch_size, patch_size,	patch_size, 1)
#    completed_region = pred_as_b012c[0, :, :, :, 0]
#    views = getviews_from_voxel_cube(completed_region,patch_size)
#    
#    
#    Y = np.load(partition + '/Y/'+ name + '.npy')
#    Y = Y[:,:,:,0]
#    Y = getviews_from_voxel_cube(Y,patch_size)
#    
#    
#    #decoded_imgs=X
#    plt.figure(figsize = (18,6))
#    for i in range(n):
#    	ax = plt.subplot(3, n, i+1)
#    	plt.imshow(X[:,:,i].reshape(patch_size,patch_size))
#    	
#    	ax.get_xaxis().set_visible(False)
#    	ax.get_yaxis().set_visible(False)
#    
#    	ax = plt.subplot(3,n, i + n+1)
#    	plt.imshow(views[:,:,i].reshape(patch_size,patch_size))
#    	
#    	ax.get_xaxis().set_visible(False)
#    	ax.get_yaxis().set_visible(False)
#    
#    	ax = plt.subplot(3,n, i + 2*n+1)
#    	plt.imshow(Y[:,:,i].reshape(patch_size,patch_size))
#    	
#    	ax.get_xaxis().set_visible(False)
#    	ax.get_yaxis().set_visible(False)
#    plt.show()
#    
#    
#    
#    #
#    voxel_cube = views_to_boxel(views,patch_size)
#    x_array, y_array, z_array = boxel_to_cloud(completed_region)
#    fig2 = plt.figure(2)
#    ax = fig2.add_subplot(111, projection='3d')
#    ax.scatter(x_array, y_array, z_array, c = 'r', marker = '.')
#    # ax.get_xaxis().set_visible(False) #RTX 2080
#    # ax.get_yaxis().set_visible(False) #RTX 2080
#    # ax.get_zaxis().set_visible(False) #RTX 2080
#    ax.xaxis.set_visible(False)
#    ax.yaxis.set_visible(False)
#    ax.zaxis.set_visible(False)
#    ax.grid(False)
#    ax.set_xticks(range(33))
#    ax.set_yticks(range(33))
#    ax.set_zticks(range(33))
#    ax.axis('off')   
#    fig2.show()
#
#    input()
#    
    
elementos = ['12873','12270','12743','12118','12989','11821','12975','12272','12545','11912','12097','12215','12468']

save = []

for img in range(decoded_imgs.shape[0]):
    name = list_IDs[img].replace('/', '_')
    if name in elementos:
         print(name)
         pred = decoded_imgs[img,:]
         pred_as_b012c = pred.reshape(1, patch_size, patch_size,	patch_size, 1)
         completed_region = pred_as_b012c[0, :, :, :, 0]
         views = getviews_from_voxel_cube(completed_region,patch_size)
         voxel_cube = views_to_boxel(views,patch_size)
         x_array, y_array, z_array = boxel_to_cloud(completed_region)
         save.append(voxel_cube)
np.save('figurasYCBVarley',save)
