from keras.models import load_model
import numpy as np
from utilities_for_data1 import views_to_boxel
from utilities_for_data1 import boxel_to_cloud
from utilities_for_data1 import dumi_voxel_cube
from mpl_toolkits.mplot3d import Axes3D
from DataGeneratorT import DataGenerator # Change to DataGeneratorT for Shapenet test
from time import gmtime, strftime

def truncate_view(view,size):
    offset = int(size*.1 + 1) 
    view = view - offset
    temp = np.logical_and(view>=size, view <= (size+4))
    view[temp] = size-1
    temp = np.logical_or(view<0,view>(size+4))
    view[temp] = -1
    view = view.astype(int)
    return view

date = strftime("%Y %m %d", gmtime())


#PC only
# import tensorflow as tf
import tensorflow.compat.v1 as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
session = tf.Session()#(config=config)

MODEL = 'YCB'
SAVE_PATH = '/home/leo/Results/'
BASE_PATH = '/home/leo/Datasets/Test_Proposal_' 

PATH = '/home/leo/Models/'
NAME = 'Model_Proposal_'+ MODEL + '.h5'
model = load_model(PATH + NAME)
    
    
path_list_IDs = BASE_PATH + MODEL + '/' + 'Test.txt'
partition = BASE_PATH + MODEL + '/Test'
params = {'dim': (32,32),
          'batch_size': 64,
          'n_channels': 6,
          'shuffle': False}


test_generator = DataGenerator(partition, path_list_IDs,**params)
decoded_imgs = model.predict_generator(generator=test_generator,
                    use_multiprocessing=True,
                    workers=16)


decoded_imgs = np.array(decoded_imgs)
tem = np.empty((decoded_imgs.shape[1],32,32,6))

for i in range(decoded_imgs.shape[1]):
    for j in range(decoded_imgs.shape[0]):
        tem[i,:,:,j] = decoded_imgs[j,i,:,:,0]


print(tem.shape)
decoded_imgs = tem


#np.save(SAVE_PATH + 'Results_Proposal_'+MODEL+'.npy',decoded_imgs)


f = open(path_list_IDs)
list_IDs = f.read().splitlines()
f.close()

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
#n = 6
#img = 210  #60 mesa  #210 carro

#
#name = list_IDs[img].replace('/', '_')
#print(name)
#X = np.load(partition + '/X/'+ name + '.npy')
#Y = np.load(partition + '/Y/'+ name + '.npy')

#decoded_imgs=X
#fig0 = plt.figure(0,figsize = (18,6))
#for i in range(n):
#    ax = plt.subplot(3, n, i+1)
#    plt.imshow(X[:,:,i].reshape(32,32))
#    
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#
#    ax = plt.subplot(3,n, i + n+1)
#    plt.imshow(truncate_view(decoded_imgs[img,:,:,i].reshape(32,32),32))
#    
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#
#    ax = plt.subplot(3,n, i + 2*n+1)
#    plt.imshow(truncate_view(Y[:,:,i].reshape(32,32),32))
#    
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
##plt.show()
#fig0.show()
#
#
#
#name = list_IDs[img].replace('/', '_')
#print(name)
#Y = np.load(partition + '/Y/'+ name + '.npy')
##voxel_cube = views_to_boxel(decoded_imgs[img,:,:,:],32)
#voxel_cube = dumi_voxel_cube(decoded_imgs[img,:,:,:],32)
#x_array, y_array, z_array = boxel_to_cloud(voxel_cube)
#plt.axis('off')    
#
#fig1 = plt.figure(1)
#ax = fig1.add_subplot(111, projection='3d')
#ax.scatter(x_array, y_array, z_array, c = 'r', marker = '.')
## ax.get_xaxis().set_visible(False) #RTX 2080
## ax.get_yaxis().set_visible(False) #RTX 2080
## ax.get_zaxis().set_visible(False) #RTX 2080
#ax.xaxis.set_visible(False)
#ax.yaxis.set_visible(False)
#ax.zaxis.set_visible(False)
#ax.grid(False)
## ax.axis('off') 
#ax.set_xticks(range(33))
#ax.set_yticks(range(33))
#ax.set_zticks(range(33))
#
#fig1.show()
#input()

elementos = ['12873','12270','12743','12118','12989','11821','12975','12272','12545','11912','12097','12215','12468']

save = []

for img in range(decoded_imgs.shape[0]):

    name = list_IDs[img].replace('/', '_')
    if name in elementos:
         print(name)
         Y = np.load(partition + '/Y/'+ name + '.npy')
    #     voxel_cube = views_to_boxel(decoded_imgs[img,:,:,:],32)
         voxel_cube = dumi_voxel_cube(decoded_imgs[img,:,:,:],32)
         x_array, y_array, z_array = boxel_to_cloud(voxel_cube)
         save.append(voxel_cube)
#         plt.axis('off')    
##    
#         fig1 = plt.figure(1)
#         ax = fig1.add_subplot(111, projection='3d')
#         ax.scatter(x_array, y_array, z_array, c = 'r', marker = '.')
#    #     ax.get_xaxis().set_visible(False)
#    #     ax.get_yaxis().set_visible(False)
#    #     ax.get_zaxis().set_visible(False)
#         ax.grid(False)
#         ax.set_xticks(range(33))
#         ax.set_yticks(range(33))
#         ax.set_zticks(range(33))
#         #fig1.axis('off')    
#         fig1.show()
#    
    
         voxel_cube = views_to_boxel(Y,32)
         #voxel_cube = dumi_voxel_cube(Y,32)
         x_array, y_array, z_array = boxel_to_cloud(voxel_cube)
         save.append(voxel_cube)
#         fig2 = plt.figure(2)
#         ax = fig2.add_subplot(111, projection='3d')
#         ax.scatter(x_array, y_array, z_array, c = 'r', marker = '.')
#    #     ax.get_xaxis().set_visible(False)
#    #     ax.get_yaxis().set_visible(False)
#    #     ax.get_zaxis().set_visible(False)
#    #     ax.grid(False)
#         ax.set_xticks(range(33))
#         ax.set_yticks(range(33))
#         ax.set_zticks(range(33))
#         #fig2.axis('off')    
#         fig2.show()
#    
#         input()
np.save('figurasYCBProposal',save)
# y = [ [x_array[i], y_array[i], z_array[i]] for i in range(x_array.shape[0])] 
# pointcloud = [x_array, y_array, z_array]

# from sklearn.cluster import DBSCAN
# clustering = DBSCAN(eps=2.5, min_samples=3).fit(y)
# labels = clustering.labels_

# objeto = []
# for i in range(x_array.shape[0]):
#     if labels[i] == 0:
#         objeto.append(y[i])
# x_array = []
# y_array = []
# z_array = []
# for i in objeto:
#     x_array.append(i[0])
#     y_array.append(i[1])
#     z_array.append(i[2])
