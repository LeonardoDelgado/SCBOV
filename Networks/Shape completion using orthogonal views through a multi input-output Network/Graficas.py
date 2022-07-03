# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 23:27:07 2019

@author: delga
"""
import numpy as np
import csv
import matplotlib.pyplot as plt
names = []
base_path = '/home/leo/Dropbox/CNN for affordances detection/models YCB 10002020 04 01/'#'/home/leo/Dropbox/CNN for affordances detection/models ShapeNet 2020 03 27/'
with open(base_path+'history.history_ycb_2020 04 01 (copy).csv') as File: #'history.history_ycb_2020 03 27 (copy).csv'
    reader = csv.reader(File, delimiter=',', quotechar=',',quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        names.append([row])
dic = {}
for elemento in names:
    print(elemento[0][0])
    elemento[0][1]=elemento[0][1][3:-1]
    elemento[0][-1]=elemento[0][-1][0:-2]
    dic[elemento[0][0]]=elemento[0][1:]

ty = 100
    
loss = np.array(dic['loss']).astype('float')[0:ty]
val_loss = np.array(dic['val_loss']).astype('float')[0:ty]

view_1 = np.array(dic['conv2d_27_loss']).astype('float')[1:ty]
view_2 = np.array(dic['conv2d_32_loss']).astype('float')[1:ty]
view_3 = np.array(dic['conv2d_37_loss']).astype('float')[1:ty]
view_4 = np.array(dic['conv2d_42_loss']).astype('float')[1:ty]
view_5 = np.array(dic['conv2d_47_loss']).astype('float')[1:ty]
view_6 = np.array(dic['conv2d_52_loss']).astype('float')[1:ty]

val_view_1 = np.array(dic['val_conv2d_27_loss']).astype('float')[1:ty]
val_view_2 = np.array(dic['val_conv2d_32_loss']).astype('float')[1:ty]
val_view_3 = np.array(dic['val_conv2d_37_loss']).astype('float')[1:ty]
val_view_4 = np.array(dic['val_conv2d_42_loss']).astype('float')[1:ty]
val_view_5 = np.array(dic['val_conv2d_47_loss']).astype('float')[1:ty]
val_view_6 = np.array(dic['val_conv2d_52_loss']).astype('float')[1:ty]

epochs = range(2, len(loss) + 1)
#plt.plot(epochs, loss, color='red', label='Training loss')
#plt.plot(epochs, val_loss, color='green', label='Validation loss')

plt.plot(epochs, view_1, color='#FF0F0F', label='Training view 1 loss')
plt.plot(epochs, view_2, color='k', label='Training view 2 loss')
plt.plot(epochs, view_3, color='y', label='Training view 3 loss')
plt.plot(epochs, view_4, color='m', label='Training view 4 loss')
plt.plot(epochs, view_5, color='c', label='Training view 5 loss')
plt.plot(epochs, view_6, color='#FF00AF', label='Training view 6 loss')

plt.plot(epochs, val_view_1, color='#FF0F0F', label='Validation view 1 loss', linestyle = ':')
plt.plot(epochs, val_view_2, color='k', label='Validation view 2 loss', linestyle = ':')
plt.plot(epochs, val_view_3, color='y', label='Validation view 3 loss', linestyle = ':')
plt.plot(epochs, val_view_4, color='m', label='Validation view 4 loss', linestyle = ':')
plt.plot(epochs, val_view_5, color='c', label='Validation view 5 loss', linestyle = ':')
plt.plot(epochs, val_view_6, color='#FF00AF', label='Validation view 6 loss', linestyle = ':')
         

#plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig(base_path+'T_V_loss_YCB.pdf', bbox_inches = 'tight')
plt.show()