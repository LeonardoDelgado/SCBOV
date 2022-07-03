# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 17:13:36 2019

@author: delga
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
#RasultadosShapeNetChamferD.npy
#RasultadosYCBChamferD.npy
#/home/leo/Datasets/RasultadosYCBChamferVarley.npy
#'/home/leo/Datasets/RasultadosShapeNetChamferD.npy'
MODEL = 'ShapeNet'
SAVE_PATH = '/home/leo/Results/'
y = np.load(SAVE_PATH + 'Results_MSE_PCN_' + MODEL  +'.npy')

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

#plt.savefig('/home/leo/Datasets/CD_proposal_ShaoeNet_dumi2_mean_scale_'+mean.__str__()+'_std_'+std.__str__()+'_.pdf')
#plt.show()
#plt.clf()




print(mean)
print(std)