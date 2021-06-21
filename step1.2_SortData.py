# -*- coding: utf-8 -*-
"""
Created on Feb 1 2021

 This code is part of the supplement materials of the submmited manuscript:
 'Physics-informed Bayesian Inference for Milling Stability Analysis'.


"""
'''Integrate data from each grid point'''
import numpy as np
import os
import scipy.io as sio

size = 800
gridNumber= 27 * 85
if not os.path.exists('GeneratedData/Step1.2Data'):
    os.makedirs('GeneratedData/Step1.2Data')

alldata = sio.loadmat('GeneratedData/Step1.1Data/all_spectral_radius-MTMcase.mat')['data']
for i in range(0, gridNumber):
    temp = []
    for j in range(0, size):
        if j == 0:
            temp = alldata[gridNumber * j + i, :]
        else:
            # print(gridNumber * j + i)
            temp = np.vstack((temp,alldata[gridNumber * j + i, :]))

    sio.savemat('GeneratedData/Step1.2Data/agentdata' + str(i) + '.mat', {'data': temp})

print("step1.2 is over, please run the step2")