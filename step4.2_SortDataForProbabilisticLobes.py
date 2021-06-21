# -*- coding: utf-8 -*-
"""
Created on Feb 1 2021

 This code is part of the supplement materials of the submmited manuscript:
 'Physics-informed Bayesian Inference for Milling Stability Analysis'.


"""
'''The sampling points for each grid point are put together'''
import numpy as np
import scipy.io as sio
import os

size = 500
gridNumber = 101 * 51
if not os.path.exists('GeneratedData/Step4.2Data'):
    os.makedirs('GeneratedData/Step4.2Data')

alldata = sio.loadmat('GeneratedData/Step4.1Data/all_spectral_radius-' + str(size) + '-forMTMcasePro.mat')['data']
for i in range(0, gridNumber):
    temp = []
    for j in range(0, size):
        if j == 0:
            temp = alldata[gridNumber * j + i, :]
        else:
            print(gridNumber * j + i)
            temp = np.vstack((temp, alldata[gridNumber * j + i, :]))

    sio.savemat('GeneratedData/Step4.2Data/agentdata' + str(i) + '.mat', {'data': temp})

print("step4.2 is over, please run the step4.3")