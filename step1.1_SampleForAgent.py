# -*- coding: utf-8 -*-
"""
Created on Feb 1 2021

 This code is part of the supplement materials of the submmited manuscript:
 'Physics-informed Bayesian Inference for Milling Stability Analysis'.

Data From:
[1]	Hajdu D, Borgioli F, Michiels W, et al. Robust stability of milling operations 
based on pseudospectral approach[J]. International Journal of Machine Tools and 
Manufacture, 2020, 149: 103516.

"""
'''Samples at each grid point for agent model'''
import sobol_seq
import os
import numpy as np
import FDM_function as MF
import time
import scipy.io as sio

# size of samples 
size = 800
if not os.path.exists('GeneratedData/Step1.1Data'):
    os.makedirs('GeneratedData/Step1.1Data')

'''
Load initial model parameters from Hajdu D
Note that the variance in muAndSigmaForAgentModel is larger than
that in muAndSigmaForOptimization for wider samples for agent model, 
A : wx, wy, cx, cy, ks, ky, kt, kr
'''
A = sio.loadmat('EssentialData/muAndSigmaForAgentModel.mat')['data']
A = np.matrix(A)

mu = np.repeat(A[:, 0].T, size, axis = 0)
sigma = np.repeat(A[:, 1].T, size, axis = 0)


# Sobol Sampling
data_ = sobol_seq.i4_sobol_generate_std_normal(8, size)
data = np.multiply(data_, sigma) + mu


# This step takes some time because of FDM calculation
# Matlab sampling can be more quick
for i in range(size):
    localtime = time.asctime(time.localtime(time.time()))
    print('Index: ', i, ' in ', size, ', Time', localtime)
    _, _, _, spectral_radius = MF.FDM(
        'SampleForAgent',
        data[i, 0],
        data[i, 1],
        data[i, 2],
        data[i, 3],
        data[i, 4],
        data[i, 5],
        data[i, 6],
        data[i, 7])

    extra_feature = np.repeat(data[i], len(spectral_radius), axis = 0)
    data_temp = np.hstack((extra_feature, spectral_radius))
    if i == 0:
        data_save = data_temp
    if i > 0:
        data_save = np.vstack((data_temp, data_save))

    ##if the calculation is complete, each step can be deleted
    sio.savemat('GeneratedData/Step1.1Data/' + str(i) + '_spectral_radius.mat', {'data': data_temp})

# this is what we need in the final step
sio.savemat('GeneratedData/Step1.1Data/all_spectral_radius-MTMcase.mat', {'data': data_save})
print("step1.1 is over, please run the step1.2")