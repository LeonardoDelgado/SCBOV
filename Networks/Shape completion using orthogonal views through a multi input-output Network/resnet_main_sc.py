from resnet_fsc  import modelo_for_shape_complation as load_MFSC
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from time import gmtime, strftime
import numpy as np
from utilities_for_data import split_views
import os
from keras.models import load_model
from sklearn.model_selection import train_test_split

#PC only
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)



Tensor_name = 'resnet2'
datasets_directory = '/media/leo/Datasets/' #Data for traing in laptop, Datasets for traing in PC
dir_data = '/home/leo/Dropbox/CNN for affordances detection/'
seed = 1

date = strftime("%Y %m %d", gmtime())
relative_name_folder = dir_data+'models laptop2 '+ date
try:
	os.mkdir(relative_name_folder)
except:
	print('The folder already exits')


np.random.seed(seed)

print('loanding data')
X = np.load(datasets_directory + 'samples_14.npy')
Y = np.load(datasets_directory + 'labels_14.npy')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

viewx_1, viewx_2, viewx_3, viewx_4, viewx_5, viewx_6 = split_views(X_train)
viewy_1, viewy_2, viewy_3, viewy_4, viewy_5, viewy_6 = split_views(y_train)

viewx_1_val, viewx_2_val, viewx_3_val, viewx_4_val, viewx_5_val, viewx_6_val = split_views(X_test)
viewy_1_val, viewy_2_val, viewy_3_val, viewy_4_val, viewy_5_val, viewy_6_val = split_views(y_test)

model = load_model('/home/leo/Dropbox/CNN for affordances detection/models 2019 08 15' + '/model-100-0.002590.h5')
try:
	model.save_weights('Datares.h5')
except:
	print('file exists do not was created')

del(model)
MFSC = load_MFSC()
MFSC.load_weights('Dataresnet.h5')

checkpoint = ModelCheckpoint(relative_name_folder+'/model-{epoch:03d}-{loss:03f}.h5',
	verbose=1,
 	monitor='val_loss',
 	save_best_only=True,
 	mode='auto') 

MFSC.compile(optimizer = 'adadelta',
	loss = ['mean_squared_error','mean_squared_error','mean_squared_error','mean_squared_error','mean_squared_error','mean_squared_error'])

epochs = 1000
batch_size = 32

history = MFSC.fit([viewx_1, viewx_2, viewx_3, viewx_4, viewx_5, viewx_6],[viewy_1, viewy_2, viewy_3, viewy_4, viewy_5, viewy_6],
	epochs = epochs,
	batch_size = batch_size,
	validation_data = ([viewx_1_val, viewx_2_val, viewx_3_val, viewx_4_val, viewx_5_val, viewx_6_val],[viewy_1_val, viewy_2_val, viewy_3_val, viewy_4_val, viewy_5_val, viewy_6_val]),
	callbacks = [TensorBoard(log_dir = '/tmp/'+Tensor_name),checkpoint])
#tensorboard --logdir=/tmp/resnet2
#http://0.0.0.0:6006

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='red', label='Training loss')
plt.plot(epochs, val_loss, color='green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

import csv
csv_file = "history.history_ycb_"+date+".csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, history.history.keys())
        writer.writeheader()
        writer.writerow(history.history)
except IOError:
    print("I/O error") 
