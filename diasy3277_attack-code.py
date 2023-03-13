
import pandas as pd
import numpy as np
import torch
import torchvision.datasets as data
import torchvision.transforms as transforms
import random
from sklearn import preprocessing
device = 'cuda'

torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
learning_rate=0.001
training_epochs=2000
batch_size=50
Scaler = preprocessing.StandardScaler()
train_data=pd.read_csv('../input/18011854kbopredicton/kbo_train.csv',header=None,skiprows=1,usecols=range(0,9))
test_data=pd.read_csv('../input/18011854kbopredicton/kbo_test.csv',header=None,skiprows=1,usecols=range(0,8))
x_train_data=train_data.loc[:,0:7]
y_train_data=train_data.loc[:,[8]]

x_train_data=np.array(x_train_data)
y_train_data=np.array(y_train_data)
x_train_data=Scaler.fit_transform(x_train_data)

x_train_data=torch.FloatTensor(x_train_data)
y_train_data=torch.FloatTensor(y_train_data)
train_dataset=torch.utils.data.TensorDataset(x_train_data,y_train_data)
data_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
linear1 = torch.nn.Linear(8, 8,bias=True)
linear2 = torch.nn.Linear(8, 16,bias=True)
linear3 = torch.nn.Linear(16, 8,bias=True)
linear4 = torch.nn.Linear(8, 1,bias=True)
relu = torch.nn.ReLU()

torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)
torch.nn.init.xavier_uniform_(linear4.weight)
#torch.nn.init.xavier_uniform_(linear5.weight)
#torch.nn.init.xavier_uniform_(linear6.weight)
#torch.nn.init.xavier_uniform_(linear7.weight)

model = torch.nn.Sequential(linear1,relu,
                            linear2,relu,
                            linear3,relu,
                            linear4
                            ).to(device)
loss=torch.nn.MSELoss().to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
total_batch=len(data_loader)
for epoch in range(2200):
  avg_cost=0

  for x, y in data_loader:
    x=x.to(device)
    y=y.to(device)

    optimizer.zero_grad()
    hypo=model(x)
    cost=loss(hypo,y)
    cost.backward()
    optimizer.step()
    avg_cost+=cost/total_batch
  if(epoch%100==0):
    print('epoch:','%04d'%(epoch),'cost=','{:.9f}'.format(avg_cost))
print('Learning finished')
with torch.no_grad():
  x_test_data=test_data.loc[:,:]
  x_test_data=np.array(x_test_data)
  x_test_data=Scaler.transform(x_test_data)
  x_test_data=torch.from_numpy(x_test_data).float().to(device)

  prediction=model(x_test_data)
correct=prediction.cpu().numpy().reshape(-1,1)
submit=pd.read_csv('../input/18011854kbopredicton/submit_sample.csv')
for i in range(len(correct)):
  submit['Expected'][i]=correct[i].item()
submit
submit.to_csv('result.csv',index=False,header=True)
