#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:30:08 2020

@author: leo


"""
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D


def single_encoder(flt = True, size = 32):
    
    chanel = 1
    flt
    int_img_1 = Input(shape = (size,size,chanel))
    layer_1_1 = Conv2D(16,(5,5), activation = 'relu', padding = 'same', trainable = flt )(int_img_1)
    layer_2_1 = MaxPooling2D((2,2), padding = 'same')(layer_1_1)
    layer_3_1 = Conv2D(16,(3,3), activation = 'relu', padding = 'same', trainable = flt )(layer_2_1)
    layer_4_1 = Conv2D(16,(3,3), activation = 'relu', padding = 'same', trainable = flt )(layer_3_1)
    layer_5_1 = Conv2D(16,(3,3), activation = 'relu', padding = 'same', trainable = flt )(layer_4_1)
    layer_6_1 = Conv2D(16,(3,3), activation = 'relu', padding = 'same', trainable = flt )(layer_5_1)
    layer_7_1 = UpSampling2D((3,3))(layer_6_1)
    layer_8_1 = Conv2D(16,(3,3), activation = 'relu',  trainable = flt )(layer_7_1)
    layer_9_1 = Conv2D(16,(3,3), activation = 'relu', trainable = flt )(layer_8_1)
    layer_10_1 = Conv2D(14,(3,3), activation = 'relu', trainable = flt )(layer_9_1)
    layer_11_1 = Conv2D(12,(3,3), activation = 'relu', trainable = flt )(layer_10_1)
    layer_12_1 = Conv2D(10,(3,3), activation = 'relu',  trainable = flt )(layer_11_1)
    layer_13_1 = Conv2D(8,(3,3), activation = 'relu', trainable = flt )(layer_12_1)
    layer_14_1 = Conv2D(8,(3,3), activation = 'relu', trainable = flt )(layer_13_1)
    layer_15_1 = Conv2D(8,(3,3), activation = 'relu', padding = 'same', trainable = flt )(layer_14_1)
    output = Conv2D(1,(3,3), activation = 'relu', trainable = flt )(layer_15_1)
    model = Model(inputs = int_img_1, outputs= output) 
    model.summary()
    return model

if __name__ == '__main__':
	MSFC =  single_encoder() #MSFC model for shape completition