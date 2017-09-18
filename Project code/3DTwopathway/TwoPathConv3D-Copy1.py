
# coding: utf-8

# In[3]:

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
        self.local_conv1 = nn.Conv3d(3, 64, (7, 7, 3))
        self.local_conv2 = nn.Conv3d(64, 64, (3, 3, 3))
        self.local_conv3 = nn.Conv3d(3, 160, (13, 13, 5))
        self.total_conv = nn.Conv3d(224, 3, (21, 21, 1))

    def forward(self, x):
        under_x = F.relu(self.local_conv3(x))
        x = self.local_conv1(x)
        x = F.max_pool3d(F.relu(x), (4, 4, 1), stride = 1)
        x = self.local_conv2(x)
        x = F.max_pool3d(F.relu(x), (2, 2, 1), stride = 1)
        x = torch.cat((x, under_x), 1)
        x = self.total_conv(x)
        x = x.view(-1,3)
        return x
        
net = TwoPathConv()
net = net.cuda(1)
print(net)

x = Variable(torch.randn(1,3,33,33,5), requires_grad = True)
x = x.cuda(1)
y_pred = net.forward(x)
print(y_pred)


# In[2]:

file_list = open('/home/yiqin/trainval-balanced.txt','r')
f = h5py.File('/home/yiqin/train.h5','r')
list1 = []
str1 = file_list.readlines()
for i in range(len(str1)):
    list1.append(str1[i][0:-1].split(' '))
print(len(list1))


# In[3]:

import pickle as pkl
input1 = open('/home/yiqin/two_path_cnn/HG0001_Val_list.pkl', 'rb')
input2 = open('/home/yiqin/two_path_cnn/training_list.pkl', 'rb')
val_list = pkl.load(input1)
train_list = pkl.load(input2)
print(len(val_list))


# In[4]:

#without multiprocessing
SAMPLE = ["LG/0001", "LG/0002", "LG/0004", "LG/0006", "LG/0008", "LG/0011",
          "LG/0012", "LG/0013", "LG/0014", "LG/0015", "HG/0001", "HG/0002",
          "HG/0003", "HG/0004", "HG/0005", "HG/0006", "HG/0007", "HG/0008",
          "HG/0009", "HG/0010", "HG/0011", "HG/0012", "HG/0013", "HG/0014",
          "HG/0015", "HG/0022", "HG/0024", "HG/0025", "HG/0026", "HG/0027"]
def create_sub_patch_phase1(size, key):
    training_patch = []
    training_label = []
    len_data = len(train_list)
    for i in range(size):
        case,x,y,z,l = train_list[key * size + i]
        x,y,z,l = int(x), int(y), int(z), int(l)
        if (l==1 or l==3 or l==4):
            l = 1
        case1 = case[0:2]
        case2 = case[3:]
        content = f[case1][case2]
        img_patch = content[:3, x-16:x+16+1, y-16:y+16+1, z-2:z+3]
        training_patch.append(img_patch)
        training_label.append(l)
    train_patch = torch.from_numpy(np.array(training_patch))
    train_label = torch.from_numpy(np.array(training_label))
    return train_patch, train_label


def create_test_patch(img = 0, x = 16, z= 100):
    patch=[]
    case = SAMPLE[img]
    case1 = case[:2]
    case2 = case[3:]
    batch = []
    _, X, Y, Z = f[case1][case2].shape
    for y in range(16, Y - 17):
        patch.append(f[case1][case2][:3, x-16:x+17, y-16:y+17, z-2:z+3])
    patch = torch.from_numpy(np.array(patch))
    return patch


def create_val_patch(size):
    val_patch = []
    val_label = []
    len_data = len(val_list)
    for i in range(size):
        case,x,y,z,l = train_list[i]
        #print(i, key)
        x,y,z,l = int(x), int(y), int(z), int(l)
        if (l==1 or l==3 or l==4):
            l = 1
        case1 = case[0:2]
        case2 = case[3:]
        content = f[case1][case2]
        img_patch = content[:3, x-16:x+16+1, y-16:y+16+1, z-2:z+3] #sample a 33x33 patch
        val_patch.append(img_patch)
        val_label.append(l)
    val_patch = torch.from_numpy(np.array(val_patch))
    val_label = torch.from_numpy(np.array(val_label))
    return val_patch, val_label


# In[5]:

prev_time = time.clock()
training_patch, training_label = create_sub_patch_phase1(512, 0)
print(training_patch.size(), training_label.size())
print(time.clock() - prev_time)


# In[6]:


prev_time = time.clock()
num_epoch = 2
batch_size = 256
iteration = len(train_list) // batch_size
net = TwoPathConv()
net = net.cuda(3)

#set hyperparams
learning_rate = 5e-4
l1_reg = 5e-5
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum = 0.9, weight_decay = 5e-10)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

#weight init
for param in net.parameters():
    ini.uniform(param, a=-5e-7, b=5e-7)

#create val set
val_patch_size = 256
val_x, val_y = create_val_patch(val_patch_size)
val_x, val_y = Variable(val_x.cuda(3)), val_y.cuda(3)

test_X = create_test_patch(img = 10)

for i in range(num_epoch):
    random.shuffle(train_list)
    for j in range(iteration):
        training_patch, training_label = create_sub_patch_phase1(batch_size, j)
        x_train, y_train = Variable(training_patch.cuda(3)), Variable(training_label.cuda(3), requires_grad=False)
        #train
        y_pred = net.forward(x_train)
        y_pred = y_pred.view(-1,3)
        loss = F.cross_entropy(y_pred, y_train)#cross entropy loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #check accuracy
        if j%2000 == 0:
            print('iteration %d /%d:'%(j, iteration), loss.data)
            print(float(j)/iteration,  'finished')
            val_pred = net.forward(val_x)
            val_pred = val_pred.view(-1, 3)
            _, predicted = torch.max(val_pred.data, 1)
            correct = (predicted == val_y).sum()
            print('Validation accuracy:', float(correct) / val_patch_size)
            print('time used:%.3f'% (time.clock() - prev_time))
    scheduler.step()
    
print ("phase1: successfully trained!")
torch.save(net.state_dict(), "/home/yiqin/3DTwopathway/phase1_TwoPathConv_net3d_3ch.txt")
print ("phase1: successfully saved!")


# In[4]:

import pickle as pkl
input1 = open('/home/yiqin/two_path_cnn/training_list_unbalanced.pkl','rb')
input2 = open('/home/yiqin/two_path_cnn/HG0001_Val_list_unbalanced.pkl', 'rb')
train_list_unbalanced = pkl.load(input1)
val_list_unbalanced = pkl.load(input2)
print(len(train_list_unbalanced))
f = h5py.File('/home/yiqin/train.h5','r')


# In[5]:

#without multiprocessing
SAMPLE = ["LG/0001", "LG/0002", "LG/0004", "LG/0006", "LG/0008", "LG/0011",
          "LG/0012", "LG/0013", "LG/0014", "LG/0015", "HG/0001", "HG/0002",
          "HG/0003", "HG/0004", "HG/0005", "HG/0006", "HG/0007", "HG/0008",
          "HG/0009", "HG/0010", "HG/0011", "HG/0012", "HG/0013", "HG/0014",
          "HG/0015", "HG/0022", "HG/0024", "HG/0025", "HG/0026", "HG/0027"]
def create_training_patch_phase2(size, key):
    training_patch = []
    training_label = []
    len_data = len(train_list_unbalanced)
    for i in range(size):
        case,x,y,z,l = train_list_unbalanced[key * size + i]
        x,y,z,l = int(x), int(y), int(z), int(l)
        case1 = case[0:2]
        case2 = case[3:]
        content = f[case1][case2]
        img_patch = content[:3, x-16:x+16+1, y-16:y+16+1, z-2:z+3] #sample a 33x33 patch
        training_patch.append(img_patch)
        training_label.append(l)
    train_patch = torch.from_numpy(np.array(training_patch))
    train_label = torch.from_numpy(np.array(training_label))
    return train_patch, train_label

def create_val_patch(size):
    val_patch = []
    val_label = []
    len_data = len(val_list_unbalanced)
    for i in range(size):
        case,x,y,z,l = val_list_unbalanced[i]
        #print(i, key)
        x,y,z,l = int(x), int(y), int(z), int(l)
        case1 = case[0:2]
        case2 = case[3:]
        content = f[case1][case2]
        img_patch = content[:3, x-16:x+16+1, y-16:y+16+1, z-2:z+3] #sample a 33x33 patch
        val_patch.append(img_patch)
        val_label.append(l)
    val_patch = torch.from_numpy(np.array(val_patch))
    val_label = torch.from_numpy(np.array(val_label))
    return val_patch, val_label


# In[6]:

num_epoch = 2
batch_size = 256
iteration = len(train_list_unbalanced) // batch_size
net = TwoPathConv()
net.load_state_dict(torch.load('/home/yiqin/3DTwopathway/phase1_TwoPathConv_net3d_3ch.txt'))
net = net.cuda(3)


learning_rate = 5e-5
l1_reg = 5e-5
optimizer = torch.optim.SGD(net.total_conv.parameters(), lr=learning_rate, momentum = 0.9, weight_decay = 5e-10)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)


# In[7]:

val_patch_size = 256
val_x, val_y = create_val_patch(val_patch_size)
print((val_y == 4).sum() *1.0 / 256)
val_x, val_y = Variable(val_x.cuda(3)), val_y.cuda(3)


# In[8]:

prev_time = time.clock()
for i in range(num_epoch):
    for j in range(iteration):
        training_patch, training_label = create_training_patch_phase2(batch_size, j)
        x_train, y_train = Variable(training_patch.cuda(3)), Variable(training_label.cuda(3), requires_grad=False)
        #train
        y_pred = net.forward(x_train)
        y_pred = y_pred.view(-1,3)
        loss = F.cross_entropy(y_pred, y_train)#cross entropy loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #check accuracy
        if j%2000 == 0:
            print('iteration %d /%d:'%(j, iteration), loss.data)
            print(float(j)/iteration,  'finished')
            val_pred = net.forward(val_x)
            val_pred = val_pred.view(-1, 3)
            _, predicted = torch.max(val_pred.data, 1)
            correct = (predicted == val_y).sum()
            print('Validation accuracy:', float(correct) / val_patch_size)
            print('time used:%.3f'% (time.clock() - prev_time))
    scheduler.step()
print ("phase2: successfully trained!")
torch.save(net.state_dict(), "/home/yiqin/3DTwopathway/phase2_TwoPathConv_net3d.txt")
print ("phase2: successfully saved!")


# In[ ]:



