from single_encoder  import single_encoder as load_MFSC
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from DataGenerator_for_encoder import DataGenerator
#from keras.models import load_model
from time import gmtime, strftime
import matplotlib.pyplot as plt
import os

#PC only
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

############################ Save directory #################################################################  
dir_data = '/home/leo/Dropbox/CNN for affordances detection/'
date = strftime("%Y %m %d", gmtime())
relative_name_folder = dir_data+'models encoder PCN YCB '+ date
try:
	os.mkdir(relative_name_folder)
except:
	print('The folder already exits')
#############################################################################################################


################################################# Directory of model ########################################
Tensor_name = 'resnet2'
#######################################################################


################################################Data generator ###############################################
#
#Dataset = 'ShapeNet'
#
#epochs = 103
#path_list_IDs = {'train':'/home/leo/Datasets/ShapeNet 32/Training.txt',
#             'validation':'/home/leo/Datasets/ShapeNet 32/Validation.txt'}
#
#partition = {'train':'/home/leo/Datasets/ShapeNet 32/Training',
#             'validation':'/home/leo/Datasets/ShapeNet 32/Validation'}
#
#params = {'dim': (32,32),
#          'batch_size': 42,
#          'n_channels': 1,
#          'shuffle': True}
#
#training_generator = DataGenerator(partition['train'], path_list_IDs['train'],**params)
#validation_generator = DataGenerator(partition['validation'],path_list_IDs['validation'], **params)
##############################################################################################################


##############################################Data generator YCB###############################################
epochs = 200
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


MFSC = load_MFSC(True)
checkpoint = ModelCheckpoint(relative_name_folder+'/model-{epoch:03d}-{loss:03f}.h5',
	verbose=1,
 	monitor='val_loss',
 	save_best_only=True,
 	mode='auto')

MFSC.compile(optimizer = 'Adam',
	loss ='mean_squared_error')

history = MFSC.fit_generator(generator=training_generator,
                   validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=16,
                    epochs = epochs,
                    callbacks = [TensorBoard(log_dir = relative_name_folder+'/'+Tensor_name),checkpoint])
#tensorboard --logdir=/tmp/resnet2
#http://0.0.0.0:6006

MFSC.save_weights(relative_name_folder+'/last_model.h5')
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
csv_file = relative_name_folder +'/history.history '+Dataset+date+".csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, history.history.keys())
        writer.writeheader()
        writer.writerow(history.history)
except IOError:
    print("I/O error") 
