from keras.models import load_model
model = load_model('/home/leo/Dropbox/CNN for affordances detection/models 2019 08 13' + '/model-098-0.110366.h5')

# array_layers = model.layers
# for elementos in array_layers:
# 	print(type(elementos))


source_indices = [24,18,12,6]

target_indices_1 = [20,14,8,2]

target_indices_2 = [19,13,7,1]

target_indices = [target_indices_1, target_indices_2]

for i, indice in enumerate(source_indices):
	weights = model.layers[-indice].get_weights()
	name_source = model.layers[-indice].name
	for target in target_indices:
		target_name = model.layers[-target[i]].name
		print('loanding weights from: '+ name_source +' into ' + target_name)
		model.layers[-target[i]].set_weights(weights)
	print('entre')
model.save_weights('Dataresnet.h5')
