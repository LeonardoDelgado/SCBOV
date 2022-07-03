from keras.models import load_model
import numpy as np
from utilities_for_data1 import views_to_boxel
from utilities_for_data1 import boxel_to_cloud
from utilities_for_data1 import dumi_voxel_cube
from mpl_toolkits.mplot3d import Axes3D
from DataGeneratorT import DataGenerator # Change to DataGeneratorT for Shapenet test
from time import gmtime, strftime
from Utilities_V1 import point_cloud_manipulation as pcm

def truncate_view(view,size):
    offset = int(size*.1 + 1) 
    view = view - offset
    temp = np.logical_and(view>=size, view <= (size+4))
    view[temp] = size-1
    temp = np.logical_or(view<0,view>(size+4))
    view[temp] = -1
    view = view.astype(int)
    return view


# def move_view(event):
#     ax.autoscale(enable=False, axis='both') 
#     koef = 8
#     zkoef = (ax.get_zbound()[0] - ax.get_zbound()[1]) / koef
#     xkoef = (ax.get_xbound()[0] - ax.get_xbound()[1]) / koef
#     ykoef = (ax.get_ybound()[0] - ax.get_ybound()[1]) / koef
#     ## Map an motion to keyboard shortcuts
#     if event.key == "ctrl+down":
#         ax.set_ybound(ax.get_ybound()[0] + xkoef, ax.get_ybound()[1] + xkoef)
#     if event.key == "ctrl+up":
#         ax.set_ybound(ax.get_ybound()[0] - xkoef, ax.get_ybound()[1] - xkoef)
#     if event.key == "ctrl+right":
#         ax.set_xbound(ax.get_xbound()[0] + ykoef, ax.get_xbound()[1] + ykoef)
#     if event.key == "ctrl+left":
#         ax.set_xbound(ax.get_xbound()[0] - ykoef, ax.get_xbound()[1] - ykoef)
#     if event.key == "down":
#         ax.set_zbound(ax.get_zbound()[0] - zkoef, ax.get_zbound()[1] - zkoef)
#     if event.key == "up":
#         ax.set_zbound(ax.get_zbound()[0] + zkoef, ax.get_zbound()[1] + zkoef)
#     # zoom option
#     if event.key == "alt+up":
#         ax.set_xbound(ax.get_xbound()[0]*0.90, ax.get_xbound()[1]*0.90)
#         ax.set_ybound(ax.get_ybound()[0]*0.90, ax.get_ybound()[1]*0.90)
#         ax.set_zbound(ax.get_zbound()[0]*0.90, ax.get_zbound()[1]*0.90)
#     if event.key == "alt+down":
#         ax.set_xbound(ax.get_xbound()[0]*1.10, ax.get_xbound()[1]*1.10)
#         ax.set_ybound(ax.get_ybound()[0]*1.10, ax.get_ybound()[1]*1.10)
#         ax.set_zbound(ax.get_zbound()[0]*1.10, ax.get_zbound()[1]*1.10)
    
#     # Rotational movement
#     elev=ax.elev
#     azim=ax.azim
#     if event.key == "shift+up":
#         elev+=10
#     if event.key == "shift+down":
#         elev-=10
#     if event.key == "shift+right":
#         azim+=10
#     if event.key == "shift+left":
#         azim-=10
        

#     ax.view_init(elev= elev, azim = azim)

#     # print which ever variable you want 

#     ax.figure.canvas.draw()

date = strftime("%Y %m %d", gmtime())


#PC only
import tensorflow as tf
config =tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
###################################ShapeNet##############################################3
PATH = '/home/leo/Dropbox/CNN for affordances detection/models ShapeNet 2020 03 22/'
NAME = 'model-059-95.761074.h5'
path_list_IDs = '/home/leo/Datasets/Test_Proposal_ShapeNet/Test.txt'
partition = '/home/leo/Datasets/Test_Proposal_ShapeNet/Test'
###########################################################################################

# ###################################YCB####################################################3
# PATH = '/home/leo/Dropbox/CNN for affordances detection/models YCB 10002020 03 29/'
# NAME = 'model-212-67.512375.h5'
# path_list_IDs = '/home/leo/Datasets/YCB 32/Test.txt'
# partition = '/home/leo/Datasets/YCB 32/Test'
############################################################################################
model = load_model(PATH + NAME)



params = {'dim': (32,32),
          'batch_size': 64,
          'n_channels': 6,
          'shuffle': False}

test_generator = DataGenerator(partition, path_list_IDs,**params)



decoded_imgs = model.predict_generator(generator=test_generator)

decoded_imgs = np.array(decoded_imgs)
tem = np.empty((decoded_imgs.shape[1],32,32,6))

for i in range(decoded_imgs.shape[1]):
    for j in range(decoded_imgs.shape[0]):
        tem[i,:,:,j] = decoded_imgs[j,i,:,:,0]


print(tem.shape)
decoded_imgs = tem


# np.save('/media/leo/Datasets/results_of_proposal_ShapeNet '+date+'.npy',decoded_imgs)


f = open(path_list_IDs)
list_IDs = f.read().splitlines()
f.close()

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('tkagg')
n = 6
img = 210  #60 mesa  #210 carro


name = list_IDs[img].replace('/', '_')
X = np.load(partition + '/X/'+ name + '.npy')
Y = np.load(partition + '/Y/'+ name + '.npy')

#decoded_imgs=X
fig0 = plt.figure(0,figsize = (18,6))
for i in range(n):
	ax = plt.subplot(3, n, i+1)
	plt.imshow(X[:,:,i].reshape(32,32))
	
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	ax = plt.subplot(3,n, i + n+1)
	plt.imshow(truncate_view(decoded_imgs[img,:,:,i].reshape(32,32),32))
	
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	ax = plt.subplot(3,n, i + 2*n+1)
	plt.imshow(truncate_view(Y[:,:,i].reshape(32,32),32))
	
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
# plt.show()
fig0.show()


voxel_cube = views_to_boxel(decoded_imgs[img,:,:,:],32)
x_array, y_array, z_array = boxel_to_cloud(voxel_cube)
#voxel_cube = dumi_voxel_cube(decoded_imgs[img,:,:,:],32)

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
plt.axis('off')	

fig1 = plt.figure(1)
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(x_array, y_array, z_array, c = 'r', marker = '.')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.get_zaxis().set_visible(False)
ax.set_xticks(range(33))
ax.set_yticks(range(33))
ax.set_zticks(range(33))
ax.axis('off')	
fig1.show()


# plt.rcParams['animation.ffmpeg_path'] = '/home/leo/anaconda3/bin/ffmpeg'
voxel_cube = views_to_boxel(Y,32)

#voxel_cube = dumi_voxel_cube(Y,32)
obj1 = boxel_to_cloud(voxel_cube)
obj1 = pcm(obj1.T)
obj1.alinear()
obj1.rotationxyz(90,axis=0)
obj1.center()
obj1 = obj1.point_cloud
x_array = obj1[:,0]
y_array = obj1[:,1]
z_array = obj1[:,2]
fig2 = plt.figure(2)
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(x_array, y_array, z_array, c = 'r', marker = '.')
ax.set_ylim(0, 33)
ax.set_xlim(0, 33)
ax.set_zlim(0, 33)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.get_zaxis().set_visible(False)
ax.set_xticks(range(33))
ax.set_yticks(range(33))
ax.set_zticks(range(33))





ax.axis('off')	
# fig2.canvas.mpl_connect("key_press_event", move_view)

plt.show()
# fig2.show()


input()



