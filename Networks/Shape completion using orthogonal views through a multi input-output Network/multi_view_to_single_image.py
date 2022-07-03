from keras.layers import Input,Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import ModelCheckpoint
from keras.models import Model 
from keras import backend as k
import numpy as np
import os
from time import gmtime, strftime
date = strftime("%Y %m", gmtime())
datasets_directory = '/media/leo/Data/' 
dir_data = '/home/leo/Desktop/'
seed = 0
size = 32
np.random.seed(seed)
porsentaje_de_entrenamiento = .8

print('loanding data')
X_train = np.load(datasets_directory + 'X_train.npy')
X_val = np.load(datasets_directory + 'X_val.npy')
print('Done')

number_of_elements = [X_train.shape[0],X_val.shape[0]]
data = [X_train,X_val]
new_data = []
for i, element in enumerate(data):
	for number in range(number_of_elements[i]):
		for view in range(6):
			new_data.append(element[number,:,:,view])

X_train = np.array(new_data)
elementos,size1,size2 = X_train.shape
X_train = X_train.reshape((elementos,size1,size2,1))

# Revolver datos
print('revolviendo')
permutation = np.random.permutation(elementos)
X_train = X_train[permutation,:,:,:]
print('done')
# split
print('separando')
split_number = int(round(porsentaje_de_entrenamiento*elementos))
X_val = X_train[split_number:,:,:]
X_train =X_train[:split_number,:,:,:]
print('done')
print(X_train.shape[0]+X_val.shape[0],'=',elementos )
print('training data: ',X_train.shape[0])
print('val data: ',X_val.shape[0])
np.save('X_train_one.npy',X_train)
np.save('X_val_one.npy',X_val)



import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')




n = 20
plt.figure(figsize = (40,4))
for i in range(n):
	ax = plt.subplot(2, n, i+1)
	plt.imshow(X_train[100+i].reshape(size,size))
	#plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	ax = plt.subplot(2,n, i + n+1)
	plt.imshow(X_train[500+n+i].reshape(size,size))
	#plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()

