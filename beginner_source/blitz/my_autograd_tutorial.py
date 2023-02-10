# %% [markdown] Autograd tutorial
# # Autograd tutorial
# My version.
#
# %%

import torch
from torchvision.models import resnet18,ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1,3,64,64)
labels = torch.rand(1,1000)

# %% [markdown]
# ## Forward Pass
# Getting predictions using Forward Pass on the pretrained model `resnet18`
# imported.
#
# %%

pred = model(data)
print(f'Predictions are \n {pred} \n')
print(f'Shape of predictions is \n {pred.shape} \n')
# %% [markdown]
# Back Propagation
# 
# %%

loss = (pred - labels).sum()
loss.backward()

opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
opt.step()

# %% [markdown]
# End of non-optional part of the tutorial.
