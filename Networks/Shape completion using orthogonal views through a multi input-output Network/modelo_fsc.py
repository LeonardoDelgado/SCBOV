import numpy as np 
from keras.models import Model
from keras.layers import Dense, concatenate, Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.utils import np_utils
from keras.models import load_model

def modelo_for_shape_complation(flt = False):
	size = 32
	chanel = 1

	#load weigths
	PATH = 'models/'
	NAME = 'model-075-0.002733.h5'
	model = load_model(PATH + NAME)
	for i in range(12):
		model.layers.pop()
	#model.summary()
	try:
		model.save_weights('Data.h5')
	except:
		print('file exists do not was created')
	del(model)

	#codificador 1
	int_img_1 = Input(shape = (size,size,chanel))
	layer_1_1 = Conv2D(16,(5,5), activation = 'relu', padding = 'same', trainable = flt )(int_img_1)
	layer_2_1 = MaxPooling2D((2,2), padding = 'same')(layer_1_1)
	layer_3_1 = Conv2D(16,(3,3), activation = 'relu', padding = 'same', trainable = flt )(layer_2_1)
	encoded_1 = Conv2D(16,(3,3), activation = 'relu', padding = 'same', trainable = flt )(layer_3_1)
	encoded_model_1 = Model(int_img_1,encoded_1)
	encoded_model_1.load_weights('Data.h5')
	#codificador 2
	int_img_2 = Input(shape = (size,size,chanel))
	layer_1_2 = Conv2D(16,(5,5), activation = 'relu', padding = 'same', trainable = flt )(int_img_2)
	layer_2_2 = MaxPooling2D((2,2), padding = 'same')(layer_1_2)
	layer_3_2 = Conv2D(16,(3,3), activation = 'relu', padding = 'same', trainable = flt )(layer_2_2)
	encoded_2 = Conv2D(16,(3,3), activation = 'relu', padding = 'same', trainable = flt )(layer_3_2)
	encoded_model_2 = Model(int_img_2,encoded_2)
	encoded_model_2.load_weights('Data.h5')
	#codificador 3
	int_img_3 = Input(shape = (size,size,chanel))
	layer_1_3 = Conv2D(16,(5,5), activation = 'relu', padding = 'same', trainable = flt )(int_img_3)
	layer_2_3 = MaxPooling2D((2,2), padding = 'same')(layer_1_3)
	layer_3_3 = Conv2D(16,(3,3), activation = 'relu', padding = 'same', trainable = flt )(layer_2_3)
	encoded_3 = Conv2D(16,(3,3), activation = 'relu', padding = 'same', trainable = flt )(layer_3_3)
	encoded_model_3 = Model(int_img_3,encoded_3)
	encoded_model_3.load_weights('Data.h5')
	#codificador 4
	int_img_4 = Input(shape = (size,size,chanel))
	layer_1_4 = Conv2D(16,(5,5), activation = 'relu', padding = 'same', trainable = flt )(int_img_4)
	layer_2_4 = MaxPooling2D((2,2), padding = 'same')(layer_1_4)
	layer_3_4 = Conv2D(16,(3,3), activation = 'relu', padding = 'same', trainable = flt )(layer_2_4)
	encoded_4 = Conv2D(16,(3,3), activation = 'relu', padding = 'same', trainable = flt )(layer_3_4)
	encoded_model_4 = Model(int_img_4,encoded_4)
	encoded_model_4.load_weights('Data.h5')
	#codificador 5
	int_img_5 = Input(shape = (size,size,chanel))
	layer_1_5 = Conv2D(16,(5,5), activation = 'relu', padding = 'same', trainable = flt )(int_img_5)
	layer_2_5 = MaxPooling2D((2,2), padding = 'same')(layer_1_5)
	layer_3_5 = Conv2D(16,(3,3), activation = 'relu', padding = 'same', trainable = flt )(layer_2_5)
	encoded_5 = Conv2D(16,(3,3), activation = 'relu', padding = 'same', trainable = flt )(layer_3_5)
	encoded_model_5 = Model(int_img_5,encoded_5)
	encoded_model_5.load_weights('Data.h5')
	#codificador 6
	int_img_6 = Input(shape = (size,size,chanel))
	layer_1_6 = Conv2D(16,(5,5), activation = 'relu', padding = 'same', trainable = flt )(int_img_6)
	layer_2_6 = MaxPooling2D((2,2), padding = 'same')(layer_1_6)
	layer_3_6 = Conv2D(16,(3,3), activation = 'relu', padding = 'same', trainable = flt )(layer_2_6)
	encoded_6 = Conv2D(16,(3,3), activation = 'relu', padding = 'same', trainable = flt )(layer_3_6)
	encoded_model_6 = Model(int_img_6,encoded_6)
	encoded_model_6.load_weights('Data.h5')

	concatenate1 = concatenate([encoded_1,encoded_2,encoded_3, encoded_4, encoded_5, encoded_6], axis = -1)
	complet_1 = Conv2D(64,(5,5), activation = 'relu', padding = 'same')(concatenate1)
	up_1 = UpSampling2D((2,2))(complet_1)
	complet_2 = Conv2D(16,(5,5), activation = 'relu', padding = 'same')(up_1)
	complet_3 = Conv2D(8,(5,5), activation = 'relu', padding = 'same')(complet_2)
	output =  Conv2D(6,(3,3), activation = 'relu', padding ='same')(complet_3)

	model = Model(inputs = [int_img_1,int_img_2,int_img_3, int_img_4, int_img_5, int_img_6], outputs = output) 
	model.summary()
	return model

if __name__ == '__main__':
	MSFC =  modelo_for_shape_complation() #MSFC model for shape completition