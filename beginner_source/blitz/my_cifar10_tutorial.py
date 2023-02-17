# %% [markdown]
# # Cifar10 Tutorial Classification
#
# - torchvision has data loaders for common image datasets.
#   - `torchvision.datasets`.
# - torchvision has data transformers for images.
#   - `torch.utils.data.DataLoader`.
#
# ## Cifar10
#
# Classes are: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`,
# `horse`, `ship`, `truck`.<br>
# Image size is: 3 x 32 x 32. 3 Color channels and 32 x 32 pixels.
#
# ### Steps
# 1. Load and normalize the CIFAR10 training and test datasets using
# `torchvision`.
# 2. Define a CNN
# 3. Define a loss function
# 4. Train the model on the training data
# 5. Test the model on the test data
#
# ### Load And Normalize The CIFAR10 data
#
# %%

import torch as T
import torchvision as Tv
import torchvision.transforms as Tr
import torch.backends as backends

if not backends.mps.is_available():
    if not backends.mps.is_built():
        print(
            "MPS not available because the current PyTorch install was not "
            "built with MPS enabled."
        )
    else:
        print(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine."
        )

else:
    device = T.device("mps")
    print(f"device is {device}")

transform = Tr.Compose([Tr.ToTensor(), Tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

train = Tv.datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
trainloader = T.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
test = Tv.datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
testloader = T.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

import matplotlib.pyplot as plt

# plt.style.use('science')
plt.ion()
import numpy as np


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(trainloader)
images, labels = next(dataiter)

imshow(Tv.utils.make_grid(images))

print(" ".join(f"{classes[labels[j]]:5s}" for j in range(batch_size)))

# %% [markdown]
# ## Define the CNN
#
#
#
# %%

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self,device=device):
        super().__init__()
        self.conv1 = nn.Conv2d( 3, 6, 5)  # input size 3, output 6, convolution 5
        self.pool = nn.MaxPool2d(2)  # square form MaxPool2d with size 2 x 2
        self.conv2 = nn.Conv2d(6, 16, 5)  # in,out,convolution
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x,device=device):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = T.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs =inputs.to(device)
        labels = labels.to(device)
        print(inputs.device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f"[{epoch + 1}, {i +1:5d}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0
print("training done")

PATH = "cifar_net.pth"
T.save(net.state_dict(), PATH)
