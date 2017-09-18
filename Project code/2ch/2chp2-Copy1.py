import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time


class TwoPathConv(nn.Module):
    def __init__(self):
        super(TwoPathConv, self).__init__()
        self.local_conv1 = nn.Conv2d(2, 64, 7)
        #self.bn1 = nn.BatchNorm2d(64)
        self.local_conv2 = nn.Conv2d(64, 64, 3)
        #self.bn2 = nn.BatchNorm2d(64)
        self.local_conv3 = nn.Conv2d(2, 160, 13)
        #self.bn3 = nn.BatchNorm2d(160)
        self.total_conv = nn.Conv2d(224, 3, 21)

    def forward(self, x):
        under_x = F.relu(self.local_conv3(x))
        x = self.local_conv1(x)
        #x = self.bn1(x)
        x = F.max_pool2d(F.relu(x), 4, stride = 1)
        x = self.local_conv2(x)
        #x = self.bn2(x)
        x = F.max_pool2d(F.relu(x), 2, stride = 1)
        x = torch.cat((x, under_x), 1)
        x = self.total_conv(x)
        x = x.view(-1,3)
        return x

import pickle as pkl
input1 = open('/home/yiqin/two_path_cnn/training_list_unbalanced.pkl','rb')
input2 = open('/home/yiqin/two_path_cnn/HG0001_Val_list_unbalanced.pkl', 'rb')
train_list_unbalanced = pkl.load(input1)
val_list_unbalanced = pkl.load(input2)
print(len(train_list_unbalanced))
print(len(val_list_unbalanced))
f = h5py.File('/home/yiqin/train.h5','r')

def create_training_patch_phase2(size, key):
    training_patch = []
    training_label = []
    len_data = len(train_list_unbalanced)
    for i in range(size):
        case,x,y,z,l = train_list_unbalanced[key * size + i]
        x,y,z,l = int(x), int(y), int(z), int(l)
        if (l==1 or l==3 or l==4):
            l = 1
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
    len_data = len(val_list_unbalanced)
    for i in range(size):
        case,x,y,z,l = val_list_unbalanced[i]
        #print(i, key)
        x,y,z,l = int(x), int(y), int(z), int(l)
        if (l==1 or l==3 or l==4):
            l = 1
        case1 = case[0:2]
        case2 = case[3:]
        content = f[case1][case2]
        img_patch = content[0:2, x-16:x+16+1, y-16:y+16+1, z] #sample a 33x33 patch
        val_patch.append(img_patch)
        val_label.append(l)
    val_patch = torch.from_numpy(np.array(val_patch))
    val_label = torch.from_numpy(np.array(val_label))
    return val_patch, val_label

#phase2 training begins
num_epoch = 2
batch_size = 512
iteration = len(train_list_unbalanced) // batch_size
net = TwoPathConv()
#net.load_state_dict(torch.load('/home/yiqin/2ch/2ch_p1_4.txt'))
net = net.cuda(3)

learning_rate = 5e-6
l1_reg = 5e-5
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum = 0.9, weight_decay = 5e-7)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

val_patch_size = 512
val_x, val_y = create_val_patch(val_patch_size)
print('Validation contains 0s :', (val_y == 0).sum() *1.0 / 512)
val_x, val_y = Variable(val_x.cuda(3)), val_y.cuda(3)
#test_X = create_test_patch(img = 10)

prev_time = time.clock()
for i in range(2):
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
        if j%10 == 0:
            print('iteration %d /%d:'%(j, iteration), loss.data)
            print(float(j)/iteration,  'finished')
            val_pred = net.forward(val_x)
            val_pred = val_pred.view(-1, 3)
            _, predicted = torch.max(val_pred.data, 1)
            correct = (predicted == val_y).sum()
            print('Validation accuracy:', float(correct) / val_patch_size)
            print('time used:%.3f'% (time.clock() - prev_time))
    scheduler.step()
    name = "/home/yiqin/2ch/2ch_p2_" + str(i) + ".txt"
    torch.save(net.state_dict(), name)
    
print ("phase2: successfully trained!")

print ("phase2: successfully saved!")