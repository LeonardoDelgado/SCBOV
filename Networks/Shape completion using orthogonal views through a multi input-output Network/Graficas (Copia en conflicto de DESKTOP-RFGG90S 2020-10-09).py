# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 23:27:07 2019

@author: delga
"""
import numpy as np
import csv
import matplotlib.pyplot as plt
names = []
with open('autoencoder2019 09 14 - copia.csv') as File:
    reader = csv.reader(File, delimiter=',', quotechar=',',quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        names.append([row])
dic = {}
for elemento in names:
    print(elemento[0][0])
    elemento[0][1]=elemento[0][1][3:-1]
    elemento[0][-1]=elemento[0][-1][0:-2]
    dic[elemento[0][0]]=elemento[0][1:]
    
    
loss = np.array(dic['loss']).astype('float')[0:152]
val_loss = np.array(dic['val_loss']).astype('float')[0:152]

view_1 = np.array(dic['conv2d_27_loss']).astype('float')[0:152]
view_2 = np.array(dic['conv2d_32_loss']).astype('float')[0:152]
view_3 = np.array(dic['conv2d_37_loss']).astype('float')[0:152]
view_4 = np.array(dic['conv2d_42_loss']).astype('float')[0:152]
view_5 = np.array(dic['conv2d_47_loss']).astype('float')[0:152]
view_6 = np.array(dic['conv2d_52_loss']).astype('float')[0:152]

val_view_1 = np.array(dic['val_conv2d_27_loss']).astype('float')[0:152]
val_view_2 = np.array(dic['val_conv2d_32_loss']).astype('float')[0:152]
val_view_3 = np.array(dic['val_conv2d_37_loss']).astype('float')[0:152]
val_view_4 = np.array(dic['val_conv2d_42_loss']).astype('float')[0:152]
val_view_5 = np.array(dic['val_conv2d_47_loss']).astype('float')[0:152]
val_view_6 = np.array(dic['val_conv2d_52_loss']).astype('float')[0:152]

epochs = range(1, len(loss) + 1)
#plt.plot(epochs, loss, color='red', label='Training loss')
#plt.plot(epochs, val_loss, color='green', label='Validation loss')

plt.plot(val_view_1[-1], view_1[-1], color='#FF0F0F', label='Branch 1', marker='o')
plt.plot(val_view_2[-1], view_2[-1], color='k', label='Branch 2', marker='o')
plt.plot(val_view_3[-1], view_3[-1], color='y', label='Branch 3', marker='o')
plt.plot(val_view_4[-1], view_4[-1], color='m', label='Branch 4', marker='o')
plt.plot(val_view_5[-1], view_5[-1], color='c', label='Branch 5 ', marker='o')
plt.plot(val_view_6[-1], view_6[-1], color='#FF00AF', label='Branch 6', marker='o')


         

#plt.title('Training and validation loss')
plt.xlabel('Validation loss')
plt.ylabel('Training Loss')
plt.legend(loc=0, borderaxespad=0.1)
plt.savefig('Degree_of_overfit', dpi = 150)
plt.show()