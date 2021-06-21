# -*- coding: utf-8 -*-
"""
Created on Feb 1 2021

 This code is part of the supplement materials of the submmited manuscript:
 'Physics-informed Bayesian Inference for Milling Stability Analysis'.


"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import FDM_function as MF
import scipy.io as sio
import pandas as pd

plt.rc('font', family = 'serif', size = 12)
matplotlib.rcParams['mathtext.fontset'] = 'cm' # Set Formula Font STIX

Title = 'David'

PrioriArgument = sio.loadmat('EssentialData/muAndSigmaForOptimization.mat')['data']

#'''The lobe diagram before optimization was calculated'''
matrix_spindle_speed1, matrix_axis_depth1, matrix_eigenvalues1, _ = MF.FDM(
    'SampleAtAllGridPoint', PrioriArgument[0, 0] * 1000, PrioriArgument[1, 0] * 1000, PrioriArgument[2, 0], PrioriArgument[3, 0],
    PrioriArgument[4, 0], PrioriArgument[5, 0], PrioriArgument[6, 0], PrioriArgument[7, 0])

'''Loading probability lobe profile data, rotation speed, cutting depth, probability'''
SS = sio.loadmat('EssentialData/SSGrid.mat')['data']
AP = sio.loadmat('EssentialData/APGrid.mat')['data']
PRO = sio.loadmat('EssentialData/ProbabilityGrid.mat')['data']

'''Load David's raw data'''
lobe_x = sio.loadmat('EssentialData/David_lobe.mat')['x'][0]
lobe_y = sio.loadmat('EssentialData/David_lobe.mat')['y'][0]
lobe_david = np.vstack((lobe_x, lobe_y)).T
lobe_david = lobe_david[np.argsort(lobe_david[:, 0])]

fig, ax = plt.subplots(figsize = (10.8, 6))

'''Probability lobes'''
factor = 1
cs = ax.contourf(SS, AP, PRO * factor, np.linspace(0, 1, 20),
                 cmap = plt.cm.GnBu,
                 alpha = 0.7,
                 linestyles = None)

cb = fig.colorbar(cs)
cb.set_ticks([0.2 * factor, 0.4 * factor, 0.6 * factor, 0.8 * factor, 1.0 * factor, ])
cb.set_ticklabels(('0.2', '0.4', '0.6', '0.8', '1.0'))
cb.set_label('Probability of Chatter')
cb.ax.tick_params(labelsize=12)

cs = ax.contour(SS, AP, PRO, [0.5], colors = 'r')
plt.clabel(cs, fontsize = 10, colors = ('k', 'r'), fmt = '%1.2f')

'''Draw a theoretical lobes diagram'''
ax.contour(matrix_spindle_speed1, matrix_axis_depth1 * 1000, matrix_eigenvalues1, [1], colors = 'k', linestyles = '-')

'''Draw the optimized picture'''
# ax.contour(matrix_spindle_speed2, matrix_axis_depth2 * 1000, matrix_eigenvalues2, [1, ], colors='r')

'''draw the results of David's experiment'''
plt.plot(lobe_david[:, 0], lobe_david[:, 1] * 1000, 'b-', label = 'David')
plt.plot([5000, 5000], [0.1, 0.1], 'k-', label = 'Original')
plt.plot([5000, 5000], [0.1, 0.1], 'r-', label = 'BSLD($p=0.5$)')

'''The experimental data'''
case5_exp = np.array(pd.read_csv('EssentialData/MTM_newCaseIncludingThePointsofUncertainty.csv', sep = ','))
markers = ['o', 'x', '^']
colors = ['k', 'k', 'w']
states = ['Stable', 'Unstable', 'Marginal']
edgecolors = ['k', 'k', 'k']
for i in range(case5_exp.shape[0]):
    plt.scatter(case5_exp[i, 0] * 10000, case5_exp[i, 1],
                color = colors[int(case5_exp[i, 2])],
                marker = markers[int(case5_exp[i, 2])],
                edgecolors = edgecolors[int(case5_exp[i, 2])], linewidths=1.5,
                s = 50)
# Prepare Marker
plt.scatter(5000, 0.25, c = colors[int(0)], marker = markers[int(0)], s = 50, label = states[int(0)])
plt.scatter(5000, 0.5, c = colors[int(1)], marker = markers[int(1)], s = 50, label = states[int(1)])
plt.scatter(5800, 0.75, c = colors[int(2)], marker = markers[int(2)], s = 50, label = states[int(2)],
            edgecolors = edgecolors[int(2)], linewidths = 1.5)

'''Training data'''
case5_train_data = np.array(pd.read_csv('EssentialData/MTM_newCase_partial.csv', sep = ','))
for i in range(case5_train_data.shape[0]):
    if i == 0:
        plt.scatter(case5_train_data[i, 0] * 10000, case5_train_data[i, 1],
                    c = '', marker = 's', s = 90, edgecolors = 'r', linewidths = 0.75, label = 'Traning Samples')
    else:
        plt.scatter(case5_train_data[i, 0] * 10000, case5_train_data[i, 1],
                    c = '', marker = 's', s = 90, edgecolors = 'r', linewidths = 0.75)

size = 12
ax.set_xlabel('Spindle speed [rev/min]')
ax.set_ylabel('Axial depth [mm]')
plt.yticks()
plt.xticks()
plt.tight_layout()
ax = plt.gca()
plt.legend(loc = 'upper left', ncol = 2)
plt.show()
plt.savefig('ResultDisplay/Fig-' + str(Title) + '.svg', format = 'svg')
plt.savefig('ResultDisplay/Fig-' + str(Title) + '.pdf', format = 'pdf')
plt.savefig('ResultDisplay/Fig-' + str(Title) + '.png', format = 'png')

print("All Over")