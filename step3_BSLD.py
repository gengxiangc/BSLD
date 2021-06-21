# -*- coding: utf-8 -*-
"""
Created on Feb 1 2021

 This code is part of the supplement materials of the submmited manuscript:
 'Physics-informed Bayesian Inference for Milling Stability Analysis'.

"""
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
from decimal import Decimal
import matplotlib
from step2_TrainAgentModel import myNN

plt.rc('font', family = 'serif', size = 10)
matplotlib.rcParams['mathtext.fontset'] = 'cm' # Set Formula Font STIX

'''Import experimental cutting data'''
case5_exp = np.array(pd.read_csv('EssentialData/MTM_newCase_partial.csv', sep = ',')) # Part of the experimental data was used for optimization
caseob_exp = np.array(pd.read_csv('EssentialData/MTM_newCase.csv', sep = ','))

'''Prior distribution'''
PrioriArgument = sio.loadmat('EssentialData/muAndSigmaForOptimization.mat')['data']

A = np.matrix(PrioriArgument)

mu_0 = A[:, 0]
sigma_0 = np.multiply(A[:, 1], np.eye(8))
mu_0 = torch.tensor(mu_0)
sigma_0 = torch.tensor(sigma_0)

# Initial Parameters
x = torch.tensor(mu_0.float(), requires_grad = True)

def likelihoodF(w):
    likelihood = -0.5 * torch.mm((w - mu_0).t(), torch.inverse(sigma_0))
    likelihood = torch.mm(likelihood, (w - mu_0))
    prior = likelihood

    for i in range(case5_exp.shape[0]):
        tempstr = 'GeneratedData/Step2Model/ss' + str(round(case5_exp[i, 0] * 10000, 2)) + \
                  'ap' +str(Decimal(case5_exp[i, 1]).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP")) + 'model.pkl'
        model = torch.load(tempstr)
        fi = 1.0 / (1.0 + torch.exp(4.0 - 4.0 * model.forward(w.t())))
        #'1.5','1.0','1.0' are the adjustable weight to punish the result
        likelihood = likelihood + 1.5 / case5_exp.shape[0] * (1.0 * case5_exp[i, 2] * torch.log(fi) + 1.0 * (1 - case5_exp[i, 2]) * torch.log(1 - fi)) 

    posterior = -likelihood
    return posterior , prior

optimizer = torch.optim.Adagrad([x,] , lr = 0.002)

era = 500
allx = x.detach().numpy()
t1, t2 = likelihoodF(x)
allposterior = t1.detach().numpy()
allprior = t2.detach().numpy()
for step in range(era):
    posterior,prior = likelihoodF(x)
    optimizer.zero_grad()
    posterior.backward(retain_graph = True)
    optimizer.step()
    allx = np.hstack((allx, x.detach().numpy()))
    allposterior = np.hstack((allposterior, posterior.detach().numpy()))
    allprior = np.hstack((allprior, prior.detach().numpy()))
    if step % 50 == 0:
        print ('step : {}, w1 = {}, w2 = {}, w3 = {}, w4 = {}, w5 = {}, w6 = {}, w7 = {}, w8 = {}, posterior = {}'.
               format(step,
                      Decimal(x.tolist()[0][0]).quantize(Decimal("0.001"), rounding = "ROUND_HALF_UP"),
                      Decimal(x.tolist()[1][0]).quantize(Decimal("0.001"), rounding = "ROUND_HALF_UP"),
                      Decimal(x.tolist()[2][0]).quantize(Decimal("0.001"), rounding = "ROUND_HALF_UP"),
                      Decimal(x.tolist()[3][0]).quantize(Decimal("0.001"), rounding = "ROUND_HALF_UP"),
                      Decimal(x.tolist()[4][0]).quantize(Decimal("0.001"), rounding = "ROUND_HALF_UP"),
                      Decimal(x.tolist()[5][0]).quantize(Decimal("0.001"), rounding = "ROUND_HALF_UP"),
                      Decimal(x.tolist()[6][0]).quantize(Decimal("0.001"), rounding = "ROUND_HALF_UP"),
                      Decimal(x.tolist()[7][0]).quantize(Decimal("0.001"), rounding = "ROUND_HALF_UP"),
                      Decimal(posterior.tolist()[0][0]).quantize(Decimal("0.001"), rounding = "ROUND_HALF_UP")))

'''The covariance matrix is calculated by Laplace approximation'''
xt = torch.tensor(x.detach().numpy(), requires_grad = True)
F, _ = likelihoodF(xt)
 # Keep the graph for calculating the second derivative
dydx = torch.autograd.grad(F, xt, create_graph=True, retain_graph=True) 

d2ydx2 = torch.tensor([])
for anygrad in dydx[0]:
    d2ydx2 = torch.cat((d2ydx2, torch.autograd.grad(anygrad, xt, retain_graph=True)[0]), 1)

d2ydx2 = d2ydx2.detach().numpy()
# The covariance matrix is obtained by Laplace approximation
Cov = np.linalg.inv(d2ydx2)  
variance = []
for i in range(8):
    variance = np.hstack((variance, Cov[i, i]))


xx = x.detach().numpy()

sio.savemat('EssentialData/posteriorCovConvergence.mat', {'parameters': allx,
                                                          'posterior' : allposterior})

sio.savemat('EssentialData/posteriorCov.mat', {'data': Cov})
sio.savemat('EssentialData/posteriorMean.mat', {'data': xx})

print("step3 is over, please run the step4.1")