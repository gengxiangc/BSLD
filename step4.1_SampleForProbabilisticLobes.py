# -*- coding: utf-8 -*-
"""
Created on Feb 1 2021

 This code is part of the supplement materials of the submmited manuscript:
 'Physics-informed Bayesian Inference for Milling Stability Analysis'.


"""
import os
import numpy as np
import FDM_function as MF
import time
import scipy.io as sio

'''Samples are taken for drawing probabilistic Lobes diagram'''
size = 500
if not os.path.exists('GeneratedData/Step4.1Data'):
    os.makedirs('GeneratedData/Step4.1Data')

postCov = sio.loadmat('EssentialData/posteriorCov.mat')['data']
postMean = sio.loadmat('EssentialData/posteriorMean.mat')['data']

postCov = 0.01 * postCov
postMean = postMean.T
postMean = postMean[0, :]
data = np.random.multivariate_normal(mean = postMean, cov = postCov, size = size)
data[:, 0:2] = data[:, 0:2] * 1000

data = np.matrix(data)

'''
Note: This step takes some time because of FDM calculation.
After obtaining posterior distribution of parameters, 
the spectral_radius can also be obtained by surrogate model.
'''
for i in range(size):
    localtime = time.asctime(time.localtime(time.time()))
    print('Index: ', i, ' in ', size, ', Time', localtime)   
    _, _, _, spectral_radius = MF.FDM(
        'SampleAtAllGridPoint',
        data[i, 0],
        data[i, 1],
        data[i, 2],
        data[i, 3],
        data[i, 4],
        data[i, 5],
        data[i, 6],
        data[i, 7])

    extra_feature = np.repeat(data[i], len(spectral_radius), axis=0)
    data_temp = np.hstack((extra_feature, spectral_radius))
    if i == 0:
        data_save = data_temp
    if i > 0:
        data_save = np.vstack((data_temp, data_save))

    ##if the calculation is complete, each step can be deleted
    sio.savemat('GeneratedData/Step4.1Data/' + str(i) + '_spectral_radius.mat', {'data': data_temp})

#this is what we need in the end
sio.savemat('GeneratedData/Step4.1Data/all_spectral_radius-' + str(size) + '-forMTMcasePro.mat', {'data': data_save})
print("step4.1 is over, please run the step4.2")