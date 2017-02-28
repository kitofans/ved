import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, input_dim=1, view_dim=320, output_dim=10):
        super(Net, self).__init__()
        self.view_dim = view_dim
        self.conv1 = nn.Conv2d(input_dim, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(view_dim, 50)
        self.fc2 = nn.Linear(50, output_dim)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.view_dim)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)

def Block(in_dim, out_dim, kernel_size, stride, pad):
    return {
            'conv': nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=pad),
            'bn': nn.BatchNorm2d(out_dim)
            }

def RegisterBlock(mod, block, name):
    mod.add_module(name+"_conv",block['conv'])


class NIN(nn.Module):
    def __init__(self):
        super(NIN, self).__init__()
        self.block0 = Block(3,192,5,1,2)
        self.block1 = Block(192,160,1,1,0)
        self.block2 = Block(160,96,1,1,0)
        self.block3 = Block(96,192,5,1,2)
        self.block4 = Block(192,192,1,1,0)
        self.block5 = Block(192,192,1,1,0)
        self.block6 = Block(192,192,3,1,1)
        self.block7 = Block(192,192,1,1,0)
        self.block8 = Block(192,10,1,1,0)
        RegisterBlock(self,self.block0,"0")
        RegisterBlock(self,self.block1,"1")
        RegisterBlock(self,self.block2,"2")
        RegisterBlock(self,self.block3,"3")
        RegisterBlock(self,self.block4,"4")
        RegisterBlock(self,self.block5,"5")
        RegisterBlock(self,self.block6,"6")
        RegisterBlock(self,self.block7,"7")
        RegisterBlock(self,self.block8,"8")

    def block_apply(self,block, x):
        return F.relu(block['bn'](block['conv'](x)))

    def forward(self, x):
        x = self.block_apply(self.block0, x)
        x = self.block_apply(self.block1, x)
        x = self.block_apply(self.block2, x)
        x = F.dropout(F.max_pool2d(x, 3, stride=2, ceil_mode=True), training=self.training)
        x = self.block_apply(self.block3, x)
        x = self.block_apply(self.block4, x)
        x = self.block_apply(self.block5, x)
        x = F.dropout(F.avg_pool2d(x, 3, stride=2, ceil_mode=True), training=self.training)
        x = self.block_apply(self.block6, x)
        x = self.block_apply(self.block7, x)
        x = self.block_apply(self.block8, x)
        x = F.avg_pool2d(x, 8, stride=1, ceil_mode=True)
        return F.log_softmax(x.view(-1, 10))
        
