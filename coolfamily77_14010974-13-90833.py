




import pandas as pd 

import numpy as np

import torch 

import torchvision.datasets as datasets

import torchvision.transforms as transforms

import torch.optim as optim

import torch.nn.functional as F

import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'



torch.manual_seed(777)

if device == 'cuda':

    torch.cuda.manual_seed_all(777)
train = pd.read_csv('../input/2020soil/2020AI_soil_train.csv')

print(train.head(10))

print(train.info())

learning_rate = 0.001

training_epoch = 1000

batch_size = 50 
x_train = train.iloc[:,1:-1]

y_train = train.iloc[:,[-1]]



x_train = np.array(x_train)

y_train = np.array(y_train)



x_train = torch.FloatTensor(x_train)

y_train = torch.FloatTensor(y_train)



print(x_train.shape)

print(y_train.shape)
train_dataset = torch.utils.data.TensorDataset(x_train,y_train)



data_loader = torch.utils.data.DataLoader(dataset = train_dataset,

                                          batch_size = batch_size,

                                          shuffle = True,

                                          drop_last=True)



linear1 = nn.Linear(7,4,bias=True)

linear2 = nn.Linear(4,4,bias=True)

linear3 = nn.Linear(4,4,bias=True)

linear4 = nn.Linear(4,4,bias=True)

linear5 = nn.Linear(4,1,bias=True)



nn.init.xavier_uniform_(linear1.weight)

nn.init.kaiming_normal_(linear2.weight)

nn.init.xavier_uniform_(linear3.weight)

nn.init.kaiming_normal_(linear4.weight)

nn.init.xavier_uniform_(linear5.weight)

relu = nn.ReLU()
model = nn.Sequential(

    linear1,relu,

    linear2,relu,

    linear3,relu,

    linear4,relu,

    linear5

).to(device)
loss = nn.MSELoss().to(device)

optimizer = optim.Adam(model.parameters(),lr=learning_rate)
total_batch = len(data_loader)



for epoch in range(training_epoch):

    avg_cost = 0

    for X,Y in data_loader:

        X = X.to(device)

        Y = Y.to(device)



        optimizer.zero_grad()

        hypothesis = model(X)

        cost = loss(hypothesis,Y)

        cost.backward()

        optimizer.step()



        avg_cost += cost/total_batch



    print('epoch {:.4f} , cost = {:.6f}'.format(epoch,avg_cost))

print('learning finished!')
test = pd.read_csv('../input/2020soil/2020_soil_test.csv')

print(test.info())

test = test.iloc[:,1:]

test = np.array(test)

test = torch.FloatTensor(test).to(device)



with torch.no_grad():

    predict = model(test)

predict
correct_prediction = predict.cpu().numpy().reshape(-1,1)

result = pd.read_csv('../input/2020soil/soil_submission.csv')
for i in range(len(correct_prediction)):

    result['Expected'][i] = correct_prediction[i]

    
result.to_csv('submit.csv',index=False)