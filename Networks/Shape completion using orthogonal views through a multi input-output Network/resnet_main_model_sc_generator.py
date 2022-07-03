from resnet_fsc_lecun_normal  import modelo_for_shape_complation as load_MFSC
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from DataGenerator import DataGenerator
from time import gmtime, strftime
import matplotlib.pyplot as plt
import numpy as np
import os

#PC only
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

 #################Semilla
seed = 56
np.random.seed(seed)

############################ Save directory #################################################################  
dir_data = '/home/leo/Dropbox/CNN for affordances detection/'
date = strftime("%Y %m %d", gmtime())
relative_name_folder = dir_data+'models YCB 1000'+ date
try:
	os.mkdir(relative_name_folder)
except:
	print('The folder already exits')
#############################################################################################################


################################################# Directory of model ########################################
Tensor_name = 'resnet3'
print('tensorboard --logdir='+relative_name_folder+'/'+Tensor_name)
#http://0.0.0.0:6006
#############################################################################################################

#
################################################Data generator ShapeNEt###############################################
#epochs = 500
#
#path_list_IDs = {'train':'/home/leo/Datasets/ShapeNet 32/Training.txt',
#             'validation':'/home/leo/Datasets/ShapeNet 32/Validation.txt'}
#
#partition = {'train':'/home/leo/Datasets/ShapeNet 32/Training',
#             'validation':'/home/leo/Datasets/ShapeNet 32/Validation'}
#
#params = {'dim': (32,32),
#          'batch_size': 64,
#          'n_channels': 6,
#          'shuffle': True}
#
#training_generator = DataGenerator(partition['train'], path_list_IDs['train'],**params)
#validation_generator = DataGenerator(partition['validation'],path_list_IDs['validation'], **params)
###############################################################################################################

##############################################Data generator YCB###############################################
epochs = 1000
path_list_IDs = {'train':'/home/leo/Datasets/YCB 32/Training.txt',
             'validation':'/home/leo/Datasets/YCB 32/Validation.txt'}

partition = {'train':'/home/leo/Datasets/YCB 32/Training',
             'validation':'/home/leo/Datasets/YCB 32/Validation'}

params = {'dim': (32,32),
          'batch_size': 64,
          'n_channels': 6,
          'shuffle': True}

training_generator = DataGenerator(partition['train'], path_list_IDs['train'],**params)
validation_generator = DataGenerator(partition['validation'],path_list_IDs['validation'], **params)
############################################################################################################
#/home/leo/Dropbox/CNN for affordances detection/models encoder PCN shape_net 2020 03 21/
#model-099-1.333640.h5
#/home/leo/Dropbox/CNN for affordances detection/models encoder PCN YCB 2020 03 29/
#model-200-1.353595.h5

MFSC = load_MFSC(flt = False,
                 load_data = True, 
                 file_name = 'model-200-1.353595.h5', 
                 PATH = '/home/leo/Dropbox/CNN for affordances detection/models encoder PCN YCB 2020 03 29/',
                 relative_name_folder = relative_name_folder)


checkpoint = ModelCheckpoint(relative_name_folder+'/model-{epoch:03d}-{loss:03f}.h5',
	verbose=1,
 	monitor='val_loss',
 	save_best_only=True,
 	mode='auto')

MFSC.compile(optimizer = 'Adam',
	loss = ['mean_squared_error','mean_squared_error','mean_squared_error','mean_squared_error','mean_squared_error','mean_squared_error'])

history = MFSC.fit_generator(generator=training_generator,
                   validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=16,
                    epochs = epochs,
                    callbacks = [TensorBoard(log_dir = relative_name_folder+'/'+Tensor_name),checkpoint])



MFSC.save_weights(relative_name_folder +'/last_model.h5')
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='red', label='Training loss')
plt.plot(epochs, val_loss, color='green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(fname= relative_name_folder+'/loss.png')
plt.show()
try:
    plt.close()
except IOError:
    print("I/O error") 

import csv
csv_file = relative_name_folder + '/'+"history.history_ycb_"+date+".csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, history.history.keys())
        writer.writeheader()
        writer.writerow(history.history)
except IOError:
    print("I/O error") 
