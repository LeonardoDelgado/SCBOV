from sklearn.model_selection import train_test_split
from reconstruction_network import get_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
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
datasets_directory = '/home/leo/Datasets/' #Data for traing in laptop, Datasets for traing in PC
dir_data = '/home/leo/Dropbox/Shape Completion Enebled Robotic Grasping/'
seed = 1

date = strftime("%Y %m %d", gmtime())
relative_name_folder = dir_data+'models laptop2 '+ date
try:
	os.mkdir(relative_name_folder)
except:
	print('The folder already exits')


np.random.seed(seed)

print('loanding data')
X = np.load(datasets_directory + 'samples_for_traing_varleys_networks40.npy')
Y = np.load(datasets_directory + 'labels_for_traing_varleys_networks40.npy')

X_trainV, X_test, y_trainV, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

X_train, X_val, y_train_tem, y_val_tem = train_test_split(X_trainV, y_trainV, test_size=0.33, random_state=42)

y_val = y_val_tem.reshape(-1,64000)
y_train = y_train_tem.reshape(-1,64000)


batch_x = np.zeros((1, 40, 40, 40, 1),
          dtype=np.float32)

batch_x2 = np.zeros((1, 40, 40, 40, 1),
          dtype=np.float32)

batch_x[0, :, :, :, 0] = y_train[0,:].reshape(40,40,40)
print(batch_x.shape)

batch_x2[0, :, :, :, 0] = y_train_tem[0,:,:,:,0]


print(batch_x2.shape)
if (np.array_equal(batch_x2,batch_x)):
	print('corret shape')
else:
	print('fail ')
	

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
#################################################################################
#################################################################################
#################Load model
#tensorboard --logdir=/tmp/Varley
#http://0.0.0.0:6006



epochs = 1000
batch_size = 32
model = get_model()
model.summary()

checkpoint = ModelCheckpoint(relative_name_folder+'/model-{epoch:03d}-{loss:03f}.h5',
	verbose=1,
 	monitor='val_loss',
 	save_best_only=True,
 	mode='auto') 


history = model.fit(X_train,y_train,
	epochs = epochs,
	batch_size = batch_size,
	validation_data = (X_val,y_val),
	callbacks = [TensorBoard(log_dir = '/tmp/'+Tensor_name),checkpoint])

model.fit()

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
csv_file = "history.history_ycb_Varley"+date+".csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, history.history.keys())
        writer.writeheader()
        writer.writerow(history.history)
except IOError:
    print("I/O error") 

np.save('/home/leo/Datasets/samples_for_test_varleys_networks'+number.__str__()+'.npy',X_test)
np.save('/home/leo/Datasets/labels_for_test_varleys_networks'+number.__str__()+'.npy',y_test)
