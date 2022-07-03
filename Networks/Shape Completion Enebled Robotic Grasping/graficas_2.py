# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 17:13:36 2019

@author: delga
"""
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import numpy as np


y = np.load("/home/leo/Datasets/RasultadosShapeNetCDVarley_scale.npy")

plt.hist(y, bins=10, alpha=1, edgecolor = 'black',  linewidth=1)
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=y.shape[0]))
#plt.gca().set_xlabel('Chamfer Distance')
plt.grid(True)
plt.rc('legend', fontsize='small')    # fontsize of the axes title
plt.rc('axes', labelsize=90)
plt.rc('xtick', labelsize='large')    # fontsize of the tick labels
plt.rc('ytick', labelsize='large')    # fontsize of the tick labels

mean = np.mean(y)
std = np.std(y)

plt.savefig('/home/leo/Datasets/CD_Varley_YCB2'+mean.__str__()+'_std_'+std.__str__()+'_.pdf')
plt.show()
plt.clf()




print(mean)
print(std)