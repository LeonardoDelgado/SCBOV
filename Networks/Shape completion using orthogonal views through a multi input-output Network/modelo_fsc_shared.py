import numpy as np 
from keras.models import Model
from keras.layers import Dense, concatenate, Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.utils import np_utils
from keras.models import load_model

def modelo_for_shape_complation():
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
	weights_list = model.get_weights()
	del(model)

	size = 32
	chanel = 1
	#codificador 1
	int_img = Input(shape = (size,size,chanel))
	layer_1 = Conv2D(16,(5,5), activation = 'relu', padding = 'same', trainable = False)(int_img)
	layer_2 = MaxPooling2D((2,2), padding = 'same')(layer_1)
	layer_3 = Conv2D(16,(3,3), activation = 'relu', padding = 'same', trainable = False)(layer_2)
	out = Conv2D(16,(3,3), activation = 'relu', padding = 'same', trainable = False)(layer_3)

	encoded = Model(int_img,out)
	encoded.load_weights('Data.h5')



	#input_1
	int_img_1 = Input(shape = (size,size,chanel))

	#input 2
	int_img_2 = Input(shape = (size,size,chanel))

	#input 3
	int_img_3 = Input(shape = (size,size,chanel))

	#input 4
	int_img_4 = Input(shape = (size,size,chanel))

	#input 5
	int_img_5 = Input(shape = (size,size,chanel))

	#input 6
	int_img_6 = Input(shape = (size,size,chanel))

	#encoded_1
	encoded_1 = encoded(int_img_1)

	#encoded 2
	encoded_2 = encoded(int_img_1)

	#encoded 3
	encoded_3 = encoded(int_img_1)

	#encoded 4
	encoded_4 = encoded(int_img_1)

	#encoded 5
	encoded_5 = encoded(int_img_1)

	#encoded 6
	encoded_6 = encoded(int_img_1)

	concatenate1 = concatenate([encoded_1,encoded_2,encoded_3, encoded_4, encoded_5, encoded_6], axis = -1)
	complet_1 = Conv2D(32,(5,5), activation = 'relu', padding = 'same')(concatenate1)
	up_1 = UpSampling2D((2,2))(complet_1)
	complet_2 = Conv2D(16,(5,5), activation = 'relu', padding = 'same')(up_1)
	complet_3 = Conv2D(8,(5,5), activation = 'relu', padding = 'same')(complet_2)
	output =  Conv2D(6,(3,3), activation = 'relu', padding ='same')(complet_3)

	model = Model(inputs = [int_img_1,int_img_2,int_img_3, int_img_4, int_img_5, int_img_6], outputs = output) 
	#model.summary()
	print('El modelo fue creado, se cargaron los pesos: '+ NAME + ' en el encoder')
	return model

if __name__ == '__main__':
	MSFC =  modelo_for_shape_complation() #MSFC model for shape completition