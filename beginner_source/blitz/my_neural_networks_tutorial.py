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
print(f'Net: {net}')
print(f'Learnable parameters: \n {list(net.parameters())} \n')
params = list(net.parameters())
print(f'Len of parameters \n {len(params)} \n')
print(f'conv1\'s weights \n {params[0].size()} \n')

# %% [markdown]
# Expected input size for this **LeNet** is 32 x 32
# 
# `net` expects a 4D tensor with dimensions:
#
# ```
# nSamples x nChannels x Height x Width
# ```
# 
# For single input sample use `input.unsqueeze(0)`.
# %%

input = torch.randn(1,1,32,32)
print(f'input: {input.shape}')
print(f'input [1,1,:4,:4] \n {input[0,0,:4,:4]} \n')

out = net(input)
print(f'output: {out}')
print(f'shape of output {out.shape}')
# %% [markdown]
#
# ```
# input -> conv2d -> ReLU -> max_pool2d -> conv2d -> ReLU -> max_pool2d ->
# flatten -> linear -> ReLU -> linear -> ReLU -> linear -> MSELoss -> loss
# ```
#
# Zero the gradient buffers of all parameters and backprops with random
# gradients:
# 
# %%

net.zero_grad()
out.backward(torch.randn(1,10))

# %% [markdown]
# ### Loss Function
# A loss function takes (output, target) pair of inputs and computes a value
# that estimates how far away the output is from the target. The `nn` module has
# several different loss functions to choose from.
#
# %%

output = net(input)
target = torch.randn(10)
target = target.view(1,-1)
ll = nn.MSELoss() # INFO: First define like this!
loss = ll(output,target) # INFO: Then compute loss
print(f'Loss is: \n {loss} \n')

print(f'loss.grad_fn \n {loss.grad_fn} \n') #MSELoss
print(f'loss.grad_fn.next_functions[0][0] \n {loss.grad_fn.next_functions[0][0]} \n') #Linear
print(f'loss.grad_fn.next_functions[0][0].... \n {loss.grad_fn.next_functions[0][0].next_functions[0][0]} \n') #ReLU

# %%
net.zero_grad()
print(f'conv1\'s bias.grad looks like before the backward function: \n {net.conv1.bias.grad} \n')
print(f'conv1\'s weight.grad looks like before the backward function: \n {net.conv1.weight.grad} \n')
loss.backward()
print(f'conv1\'s bias.grad looks like after the backward function: \n {net.conv1.bias.grad} \n')
print(f'conv1\'s weight.grad looks like after the backward function: \n {net.conv1.weight.grad} \n')
print(f'conv2\'s weight.grad looks like after the backward function: \n {net.conv2.weight.grad} \n')

# %% [markdown]
# ### Updating The Weights
# 
#    weight = weight - lr * gradient
# %%
lr = 1e-2
for p in net.parameters():
    p.data.sub_(p.grad.data * lr)

# %% [markdown]
# ### Using torch.optim for optimization
# 
# 
#
# %%

import torch.optim as optim
# Creation
optimizer = optim.SGD(net.parameters(), lr=1e-2)

# Application
optimizer.zero_grad() # zero gradient buffers
out = net(input)
criterion = nn.MSELoss()
loss = criterion(out,target)
loss.backward()
optimizer.step() # Update step



