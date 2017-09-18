
# coding: utf-8

# In[2]:

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
import torch.nn.init as ini
import multiprocessing
from multiprocessing import Queue
import random

class TwoPathConv(nn.Module):
    def __init__(self):
        super(TwoPathConv, self).__init__()
        self.local_conv1 = nn.Conv2d(4, 64, 7)
        self.dp1 = nn.Dropout2d(.3)
        self.local_conv2 = nn.Conv2d(64, 64, 3)
        self.dp2 = nn.Dropout2d(.3)
        self.local_conv3 = nn.Conv2d(4, 160, 13)
        self.dp3 = nn.Dropout2d(.3)
        self.total_conv = nn.Conv2d(224, 5, 21)

    def forward(self, x):
        under_x = F.relu(self.dp3(self.local_conv3(x)))
        x = self.local_conv1(x)
        x = self.dp1(x)
        x = F.max_pool2d(F.relu(x), 4, stride = 1)
        x = self.local_conv2(x)
        x = self.dp2(x)
        x = F.max_pool2d(F.relu(x), 2, stride = 1)
        x = torch.cat((x, under_x), 1)
        x = self.total_conv(x)
        x = x.view(-1,5)
        return x

# In[3]:

file_list = open('/home/yiqin/trainval-balanced.txt','r')
f = h5py.File('/home/yiqin/train.h5','r')
list1 = []
str1 = file_list.readlines()
for i in range(len(str1)):
    list1.append(str1[i][0:-1].split(' '))
print(len(list1))

import pickle as pkl
input1 = open('HG0001_Val_list.pkl', 'rb')
input2 = open('training_list.pkl', 'rb')
val_list = pkl.load(input1)
train_list = pkl.load(input2)
print(len(val_list))


#without multiprocessing
def create_sub_patch_phase1(size, key):
    training_patch = []
    training_label = []
    len_data = len(train_list)
    for i in range(size):
        case,x,y,z,l = train_list[key * size + i]
        x,y,z,l = int(x), int(y), int(z), int(l)
        case1 = case[0:2]
        case2 = case[3:]
        content = f[case1][case2]
        img_patch = content[:, x-16:x+16+1, y-16:y+16+1, z] #sample a 33x33 patch
        training_patch.append(img_patch)
        training_label.append(l)
    train_patch = torch.from_numpy(np.array(training_patch))
    train_label = torch.from_numpy(np.array(training_label))
    return train_patch, train_label


def create_val_patch(size):
    val_patch = []
    val_label = []
    len_data = len(val_list)
    for i in range(size*2, size*3):
        case,x,y,z,l = val_list[i]
        #print(i, key)
        x,y,z,l = int(x), int(y), int(z), int(l)
        case1 = case[0:2]
        case2 = case[3:]
        content = f[case1][case2]
        img_patch = content[:, x-16:x+16+1, y-16:y+16+1, z] #sample a 33x33 patch
        val_patch.append(img_patch)
        val_label.append(l)
    val_patch = torch.from_numpy(np.array(val_patch))
    val_label = torch.from_numpy(np.array(val_label))
    return val_patch, val_label


prev_time = time.clock()
num_epoch = 5
batch_size = 512
iteration = len(train_list) // batch_size
net = TwoPathConv()
#net.load_state_dict(torch.load('/home/yiqin/phase1_TwoPathConv_net.txt'))
net = net.cuda(3)

#net init
'''
for param in net.parameters():
    if len(param.size())==4:
        ini.uniform(param, a=-5e-3, b=5e-3)
'''        
#set hyperparams
learning_rate = 5e-4
l1_reg = 5e-7
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum = 0.9, weight_decay = 5e-7)
scheduler = StepLR(optimizer, step_size=1, gamma=0.2)
#create val set
val_patch_size = 512
val_x, val_y = create_val_patch(val_patch_size)
val_x, val_y = Variable(val_x.cuda(3)), val_y.cuda(3)

for i in range(num_epoch):
    for j in range(iteration):
        training_patch, training_label = create_sub_patch_phase1(batch_size, j)
        x_train, y_train = Variable(training_patch.cuda(3)), Variable(training_label.cuda(3), requires_grad=False)
        #train
        y_pred = net.forward(x_train)
        y_pred = y_pred.view(-1,5)
        loss = F.cross_entropy(y_pred, y_train)#cross entropy loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #check accuracy
        if j%10 == 0:
            print('iteration %d /%d:'%(j, iteration), loss.data)
            print(float(j)/iteration,  'finished')
            val_pred = net.forward(val_x)
            val_pred = val_pred.view(-1, 5)
            _, predicted = torch.max(val_pred.data, 1)
            correct = (predicted == val_y).sum()
            print('Validation accuracy:', float(correct) / val_patch_size)
            print('time used:%.3f'% (time.clock() - prev_time))
    scheduler.step()
    output = "/home/yiqin/two_path_cnn/phase1_TwoPathConv_net" + str(i) + '.txt'
    torch.save(net.state_dict(), output)
    
print ("phase1: successfully trained!")

print ("phase1: successfully saved!")

