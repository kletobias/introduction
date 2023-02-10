# %% [markdown]
# # Neural Networks Tutorial
# My version of the tutorial around the `nn` module in the pytorch library.
# 
# ## Define The Network
# %%

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5) # output is 16 channels
        # y = Wx + b
        self.fc1 = nn.Linear(16*5*5,120) # TODO: is 120 the bias scalar? Is first parameter Wx?
        self.fc2 = nn.Linear(120,84) # second parameter is output size
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2)) # TODO: understand
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # INFO: case square size
        x = torch.flatten(x,1) # only keep batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)
