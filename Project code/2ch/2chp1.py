
# coding: utf-8

# In[1]:

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
import random

class TwoPathConv(nn.Module):
    def __init__(self):
        super(TwoPathConv, self).__init__()
        self.local_conv1 = nn.Conv2d(2, 64, 7)
        self.local_conv2 = nn.Conv2d(64, 64, 3)
        self.local_conv3 = nn.Conv2d(2, 160, 13)
        self.total_conv = nn.Conv2d(224, 3, 21)

    def forward(self, x):
        under_x = F.relu(self.local_conv3(x))
        x = self.local_conv1(x)
        x = F.max_pool2d(F.relu(x), 4, stride = 1)
        x = self.local_conv2(x)
        x = F.max_pool2d(F.relu(x), 2, stride = 1)
        x = torch.cat((x, under_x), 1)
        x = self.total_conv(x)
        x = x.view(-1,3)
        return x


# In[3]:

f = h5py.File('/home/yiqin/train.h5','r')

import pickle as pkl
input1 = open('/home/yiqin/two_path_cnn/HG0001_Val_list.pkl', 'rb')
input2 = open('2ch_train.pkl', 'rb')
val_list = pkl.load(input1)
train_list = pkl.load(input2)
print(len(val_list))


# In[4]:

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
        img_patch = content[0:2, x-16:x+16+1, y-16:y+16+1, z] #sample a 33x33 patch
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
        img_patch = content[0:2, x-16:x+16+1, y-16:y+16+1, z] #sample a 33x33 patch
        val_patch.append(img_patch)
        val_label.append(l)
    val_patch = torch.from_numpy(np.array(val_patch))
    val_label = torch.from_numpy(np.array(val_label))
    return val_patch, val_label


# In[5]:

prev_time = time.clock()
num_epoch = 3
batch_size = 512
iteration = len(train_list) // batch_size
net = TwoPathConv()

#net init
for param in net.parameters():
    if len(param.size())==4:
        ini.uniform(param, a=-5e-3, b=5e-3)
        
#net.load_state_dict(torch.load('2ch_p1_4.txt'))
net = net.cuda(0)


#set hyperparams
learning_rate = 1e-3
l1_reg = 5e-7
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum = 0.9, weight_decay = 5e-7)
scheduler = StepLR(optimizer, step_size=1, gamma=0.2)
#create val set
val_patch_size = 512
val_x, val_y = create_val_patch(val_patch_size)
val_x, val_y = Variable(val_x.cuda(0)), val_y.cuda(0)

for i in range(num_epoch):
    random.shuffle(train_list)
    for j in range(iteration):
        training_patch, training_label = create_sub_patch_phase1(batch_size, j)
        x_train, y_train = Variable(training_patch.cuda(0)), Variable(training_label.cuda(0), requires_grad=False)
        #train
        y_pred = net.forward(x_train)
        y_pred = y_pred.view(-1,3)
        loss = F.cross_entropy(y_pred, y_train)#cross entropy loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #check accuracy
        if j%10 == 0:
            print('iteration %d /%d:'%(j, iteration), loss.data)
            print(float(j)/(iteration * num_epoch),  'finished')
            val_pred = net.forward(val_x)
            val_pred = val_pred.view(-1, 3)
            _, predicted = torch.max(val_pred.data, 1)
            correct = (predicted == val_y).sum()
            print('Validation accuracy:', float(correct) / val_patch_size)
            print('time used:%.3f'% (time.clock() - prev_time))
    scheduler.step()
    output = "2ch_p1_" + str(i) + '.txt'
    torch.save(net.state_dict(), output)
    
print ("phase1: successfully trained!")

print ("phase1: successfully saved!")


# In[ ]:



