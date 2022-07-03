from keras.models import Model
from keras.layers import concatenate, Input, Conv2D, MaxPooling2D, UpSampling2D
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
	complet_1 = Conv2D(32,(5,5), activation = 'selu', padding = 'same')(concatenate1)
	up_1 = UpSampling2D((2,2))(complet_1)
	complet_2 = Conv2D(16,(5,5), activation = 'selu', padding = 'same')(up_1)
	complet_3 = Conv2D(16,(5,5), activation = 'selu', padding = 'same')(complet_2)
	complet_4 =  Conv2D(16,(3,3), activation = 'selu', padding ='same')(complet_3)


	view_1_1 = UpSampling2D((2,2))(encoded_1)
	convv1_1 = Conv2D(16,(3,3), activation = 'selu', padding = 'same')(view_1_1)
	conc_1_1 = concatenate([complet_4,convv1_1], axis = -1)
	convv1_2 = Conv2D(32,(3,3), activation = 'selu', padding = 'same')(conc_1_1)
	convv1_3 = Conv2D(16,(5,5), activation = 'selu', padding = 'same')(convv1_2)
	convv1_4 = Conv2D(4,(5,5), activation = 'selu', padding = 'same')(convv1_3)
	cout_1_1 = Conv2D(1,(3,3), activation = 'relu', padding = 'same')(convv1_4)

	view_2_1 = UpSampling2D((2,2))(encoded_2)
	convv2_1 = Conv2D(16,(3,3), activation = 'selu', padding = 'same')(view_2_1)
	conc_2_1 = concatenate([complet_4,convv2_1], axis = -1)
	convv2_2 = Conv2D(32,(3,3), activation = 'selu', padding = 'same')(conc_2_1)
	convv2_3 = Conv2D(16,(5,5), activation = 'selu', padding = 'same')(convv2_2)
	convv2_4 = Conv2D(4,(5,5), activation = 'selu', padding = 'same')(convv2_3)
	cout_2_1 = Conv2D(1,(3,3), activation = 'relu', padding = 'same')(convv2_4)

	view_3_1 = UpSampling2D((2,2))(encoded_3)
	convv3_1 = Conv2D(16,(3,3), activation = 'selu', padding = 'same')(view_3_1)
	conc_3_1 = concatenate([complet_4,convv3_1], axis = -1)
	convv3_2 = Conv2D(32,(3,3), activation = 'selu', padding = 'same')(conc_3_1)
	convv3_3 = Conv2D(16,(5,5), activation = 'selu', padding = 'same')(convv3_2)
	convv3_4 = Conv2D(4,(5,5), activation = 'selu', padding = 'same')(convv3_3)
	cout_3_1 = Conv2D(1,(3,3), activation = 'relu', padding = 'same')(convv3_4)

	view_4_1 = UpSampling2D((2,2))(encoded_4)
	convv4_1 = Conv2D(16,(3,3), activation = 'selu', padding = 'same')(view_4_1)
	conc_4_1 = concatenate([complet_4,convv4_1], axis = -1)
	convv4_2 = Conv2D(32,(3,3), activation = 'selu', padding = 'same')(conc_4_1)
	convv4_3 = Conv2D(16,(5,5), activation = 'selu', padding = 'same')(convv4_2)
	convv4_4 = Conv2D(4,(5,5), activation = 'selu', padding = 'same')(convv4_3)
	cout_4_1 = Conv2D(1,(3,3), activation = 'relu', padding = 'same')(convv4_4)

	view_5_1 = UpSampling2D((2,2))(encoded_5)
	convv5_1 = Conv2D(16,(3,3), activation = 'selu', padding = 'same')(view_5_1)
	conc_5_1 = concatenate([complet_4,convv5_1], axis = -1)
	convv5_2 = Conv2D(32,(3,3), activation = 'selu', padding = 'same')(conc_5_1)
	convv5_3 = Conv2D(16,(5,5), activation = 'selu', padding = 'same')(convv5_2)
	convv5_4 = Conv2D(4,(5,5), activation = 'selu', padding = 'same')(convv5_3)
	cout_5_1 = Conv2D(1,(3,3), activation = 'relu', padding = 'same')(convv5_4)

	view_6_1 = UpSampling2D((2,2))(encoded_6)
	convv6_1 = Conv2D(16,(3,3), activation = 'selu', padding = 'same')(view_6_1)
	conc_6_1 = concatenate([complet_4,convv6_1], axis = -1)
	convv6_2 = Conv2D(32,(3,3), activation = 'selu', padding = 'same')(conc_6_1)
	convv6_3 = Conv2D(16,(5,5), activation = 'selu', padding = 'same')(convv6_2)
	convv6_4 = Conv2D(4,(5,5), activation = 'selu', padding = 'same')(convv6_3)
	cout_6_1 = Conv2D(1,(3,3), activation = 'relu', padding = 'same')(convv6_4)


	model = Model(inputs = [int_img_1,int_img_2,int_img_3, int_img_4, int_img_5, int_img_6], outputs = [cout_1_1, cout_2_1,cout_3_1,cout_4_1,cout_5_1,cout_6_1]) 
	model.summary()
	return model

if __name__ == '__main__':
	MSFC =  modelo_for_shape_complation() #MSFC model for shape completition
