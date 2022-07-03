from keras.models import load_model
import numpy as np
from utilities_for_data import split_views
from sklearn.model_selection import train_test_split

datasets_directory = '/media/leo/Datasets/' 
PATH = 'models laptop2 2019 08 27/'
NAME = 'model-502-2.816164.h5'
model = load_model(PATH + NAME)

X = np.load(datasets_directory + 'samples_14.npy')
Y = np.load(datasets_directory + 'labels_14.npy')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

#viewx_1, viewx_2, viewx_3, viewx_4, viewx_5, viewx_6 = split_views(X_train)
#viewy_1, viewy_2, viewy_3, viewy_4, viewy_5, viewy_6 = split_views(y_train)
del(X_train,y_train)

viewx_1_val, viewx_2_val, viewx_3_val, viewx_4_val, viewx_5_val, viewx_6_val = split_views(X_test)
viewy_1_val, viewy_2_val, viewy_3_val, viewy_4_val, viewy_5_val, viewy_6_val = split_views(y_test)
print(viewx_1_val.shape)
print(y_test.shape)


decoded_imgs =  model.predict([viewx_1_val, viewx_2_val, viewx_3_val, viewx_4_val, viewx_5_val, viewx_6_val])
decoded_imgs = np.array(decoded_imgs)
print(decoded_imgs.shape)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

n = 6
img = 60
plt.figure(figsize = (18,6))
for i in range(n):
	ax = plt.subplot(3, n, i+1)
	plt.imshow(X_test[img,:,:,i].reshape(32,32))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	ax = plt.subplot(3,n, i + n+1)
	plt.imshow(decoded_imgs[i,img,:,:,:].reshape(32,32))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	ax = plt.subplot(3,n, i + 2*n+1)
	plt.imshow(y_test[img,:,:,i].reshape(32,32))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()
