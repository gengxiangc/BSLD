# -*- coding: utf-8 -*-
"""
Created on Feb 1 2021

 This code is part of the supplement materials of the submmited manuscript:
 'Physics-informed Bayesian Inference for Milling Stability Analysis'.

Note:
    the AgentModel can be trained by grid, or by one big model
       
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import scipy.io as scio
import torch.optim as optim
import os
from decimal import Decimal

size = 800
gridNumber = 27 * 85

class myNN(torch.nn.Module):
    def __init__(self):
        super(myNN, self).__init__()
        self.fc1 = nn.Linear(8, 20)
        self.fc2 = nn.Linear(20, 40)
        self.fc3 = nn.Linear(40, 40)
        self.fc4 = nn.Linear(40, 1)
        self.y_prediction = None

    def forward(self,x):
        h1 = F.sigmoid(self.fc1(x))
        h2 = F.sigmoid(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        self.y_prediction = F.relu(self.fc4(h3))
        return self.y_prediction

    def loss(self,y):
        return torch.norm(self.y_prediction - y) / y.size()[0]

if __name__=="__main__":
    if not os.path.exists('GeneratedData/Step2Model'):
        os.makedirs('GeneratedData/Step2Model')

    for i in range(gridNumber):
        print('Training model : ', i, ' in ', gridNumber, 'grid')
        temp = scio.loadmat('GeneratedData/Step1.2Data/agentdata' + str(i) + '.mat')['data']
        temp[:, 0:2] = temp[:, 0:2] / 1000  
        #In order to facilitate neural network training, the natural frequency is reduced by 1000 
        x = torch.tensor(temp[:, 0:8], dtype = torch.float32)
        ss = temp[0, 8]
        ap = Decimal(temp[0, 9] * 1000).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP")
        y = torch.tensor(temp[:, 10], dtype = torch.float32)
        y = y.resize(size, 1)

        tempmodel = myNN()
        optimizer = optim.Adam(tempmodel.parameters(), lr = 0.01)
        ecohp = 0
        while(ecohp < 3000 ):
            tempmodel.forward(x)
            loss = tempmodel.loss(y)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ecohp += 1

        torch.save(tempmodel, 'GeneratedData/Step2Model/ss' + str(ss) + 'ap'+str(ap)+'model.pkl')

    print("step2 is over, please run the step3")