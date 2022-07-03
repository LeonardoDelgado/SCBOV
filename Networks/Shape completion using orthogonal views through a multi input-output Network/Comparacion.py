import numpy as np
from etxt import getviews_from_voxel_cube
from utilities_for_data import views_to_boxel
from utilities_for_data import boxel_to_cloud

def ercm(views1,views2):
	views = views1.shape[2]
	elementos = views1.shape[0]*views1.shape[1]
	results = []
	for view in range(views):
		results.append(np.sum((views1[:,:,view]-views2[:,:,view])**2)/elementos)
	results = np.sum(results)/6
	return results


patch_size = 32
datasets_directory = '/media/leo/Datasets/'
y = np.load(datasets_directory + 'labels_18_14.npy')
x = np.load(datasets_directory + 'results_of_proposal.npy')

elementos = y.shape[0]
resultados = []
for elemento in range(elementos):
	print(elemento)
	views = x[elemento,:,:,:]
	views2 = y[elemento,:,:,:]
	results = ercm(views,views2)
	resultados.append(results)
	print('Resultados 3, ', results)

resultados = np.array(resultados)

np.save('/media/leo/Datasets/resultados_proposal.npy',resultados)


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
n = 6
plt.figure(figsize = (12,6))
for i in range(n):
	ax = plt.subplot(2, n, i+1)
	plt.imshow(views[:,:,i].reshape(patch_size,patch_size))
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	ax = plt.subplot(2,n, i + n+1)
	plt.imshow(views2[:,:,i].reshape(patch_size,patch_size))
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)


plt.show()

voxel_cube = views_to_boxel(views,patch_size)
x_array, y_array, z_array = boxel_to_cloud(voxel_cube)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_array, y_array, z_array, c = 'r', marker = '.')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.get_zaxis().set_visible(False)
	
plt.show()

