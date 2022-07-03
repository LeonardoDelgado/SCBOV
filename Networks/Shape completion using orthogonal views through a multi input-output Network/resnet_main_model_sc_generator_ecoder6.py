from resnet_fsc_lecun_normal  import modelo_for_shape_complation as load_MFSC
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from DataGenerator_encoder6 import DataGenerator
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
relative_name_folder = dir_data+'models laptop2 ecoder6 '+ date
try:
	os.mkdir(relative_name_folder)
except:
	print('The folder already exits')
#############################################################################################################


################################################# Directory of model ########################################
Tensor_name = 'resnet26'
datasets_directory = '/media/leo/Datasets/' 
PATH = 'models encoder PCN laptop2 2020 02 17/'
NAME = 'model-097-1.472225.h5'
#############################################################################################################


###############################################Data generator ###############################################
epochs = 103
path_list_IDs = {'train':'/home/leo/Dropbox/CNN for affordances detection/Training_shapenet_32.txt',
             'validation':'/home/leo/Dropbox/CNN for affordances detection/Validation_shapenet_32.txt'}

partition = {'train':'/home/leo/Datasets/Training_32/Training',
             'validation':'/home/leo/Datasets/Training_32/Validation'}

params = {'dim': (32,32),
          'batch_size': 64,
          'n_channels': 6,
          'shuffle': True}

training_generator = DataGenerator(partition['train'], path_list_IDs['train'],**params)
validation_generator = DataGenerator(partition['validation'],path_list_IDs['validation'], **params)
#############################################################################################################


#model = load_model(PATH + NAME)
#try:
#	model.save_weights('Dataresnet_2.h5')
#except:
#	print('file exists do not was created')
    
#del(model)


MFSC = load_MFSC(flt = False,load_data = True,PATH = PATH, file_name = NAME, relative_name_folder = relative_name_folder)
#MFSC.load_weights('Dataresnet_2.h5')
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
                    callbacks = [TensorBoard(log_dir = '/tmp/'+Tensor_name),checkpoint])
#tensorboard --logdir=/tmp/resnet26
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
csv_file = relative_name_folder = relative_name_folder +'/'+"history.history_ycb_"+date+".csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, history.history.keys())
        writer.writeheader()
        writer.writerow(history.history)
except IOError:
    print("I/O error") 
