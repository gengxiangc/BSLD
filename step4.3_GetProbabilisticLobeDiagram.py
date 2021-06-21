# -*- coding: utf-8 -*-
"""
Created on Feb 1 2021

 This code is part of the supplement materials of the submmited manuscript:
 'Physics-informed Bayesian Inference for Milling Stability Analysis'.


"""

'''Monte Carlo'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

size = 500

PRO = np.zeros((51, 101))
ss = np.linspace(4800, 13200, 101)
ap = np.linspace(0, 4, 51)
SS,AP = np.meshgrid(ss, ap)

# Chatter if spectral radius is larger than 1.
for i in range(101):
    for j in range(51):
        sum = 0
        temp = scio.loadmat('GeneratedData/Step4.2Data/agentdata' + str(i*51 + j) + '.mat')['data']
        for k in range(size):
            if temp[k, 10] > 1:
                sum = sum + 1
        PRO[j, i] = sum / size
    print(i)

plt.contourf(SS, AP, PRO)

plt.contour(SS, AP, PRO)

contour = plt.contour(SS, AP, PRO, [0.5] , colors = 'k')

plt.clabel(contour, fontsize=10, colors = ('k', 'r'))

case5_exp = np.array(pd.read_csv('EssentialData/MTM_newCase.csv', sep=','))
markers = ['o', 'x', 'o']
colors  = ['y', 'r', 'b']
for i in range(case5_exp.shape[0]):
    plt.scatter(case5_exp[i, 0] * 10000, case5_exp[i, 1], c = colors[int(case5_exp[i, 2])], marker = markers[int(case5_exp[i, 2])], s = 30)

plt.show()

scio.savemat('EssentialData/SSGrid.mat', {'data': SS})
scio.savemat('EssentialData/APGrid.mat', {'data': AP})
scio.savemat('EssentialData/ProbabilityGrid.mat', {'data': PRO})

print("step4.3 is over, please run the step5_final")