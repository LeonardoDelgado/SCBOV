from sklearn.model_selection import train_test_split
from reconstruction_network import get_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from DataGenerator_Varley import DataGenerator
from time import gmtime, strftime
import matplotlib.pyplot as plt
import curvox.pc_vox_utils
import numpy as np
import os



#PC only
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)



Tensor_name = 'Varley'
dir_data = '/home/leo/Dropbox/Shape Completion Enebled Robotic Grasping/'
seed = 1

date = strftime("%Y %m %d", gmtime())
relative_name_folder = dir_data+'models laptop2 YCB'+ date
try:
	os.mkdir(relative_name_folder)
except:
	print('The folder already exits')


np.random.seed(seed)




	


#################################################################################
#'Training_shapenet_Varley.txt',
#'Validation_shapenet_Varley.txt'

#Training_YCB_Varley.txt
#Validation_YCB_Varley.txt

path_list_IDs = {'train':dir_data+'Training_YCB_Varley.txt',
             'validation':dir_data+'Validation_YCB_Varley.txt'}

#/home/leo/Datasets/YCB_source/Validation_YCB_Varley
#'/home/leo/Datasets/Training_Varley/Validation'

#/home/leo/Datasets/YCB_source/Training_YCB_Varley
#/home/leo/Datasets/Training_Varley/Training

partition = {'train':'/home/leo/Datasets/YCB_source/Training_YCB_Varley',
             'validation':'/home/leo/Datasets/YCB_source/Validation_YCB_Varley'}

params = {'dim': (40,40,40),
          'batch_size': 16,
          'n_channels': 1,
          'shuffle': True}

training_generator = DataGenerator(partition['train'], path_list_IDs['train'],**params)
validation_generator = DataGenerator(partition['validation'],path_list_IDs['validation'], **params)


#################################################################################
#################Load model
#tensorboard --logdir=/home/leo/Dropbox/Shape Completion Enebled Robotic Grasping/Varley
#http://0.0.0.0:6006



epochs = 103
model = get_model()
model.summary()

checkpoint = ModelCheckpoint(relative_name_folder+'/model-{epoch:03d}-{loss:03f}.h5',
	verbose=103,
 	monitor='val_loss',
 	save_best_only=True,
 	mode='auto') 


history = model.fit_generator(generator=training_generator,
                   validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=16,
                    epochs = epochs,
                    callbacks = [TensorBoard(log_dir = dir_data +Tensor_name),checkpoint])
#tensorboard --logdir=/tmp/resnet3
#http://0.0.0.0:6006

model.save_weights(relative_name_folder+'/last_model.h5')
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
csv_file = relative_name_folder + '/'+"history.history_ycb_"+date+".csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, history.history.keys())
        writer.writeheader()
        writer.writerow(history.history)
except IOError:
    print("I/O error") 
