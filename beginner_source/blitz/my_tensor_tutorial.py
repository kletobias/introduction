# %% [markdown]
# my version tensor tutorial
#
# %%

import torch
import numpy as np

data = [[2, 8], [3, 5]]  # list
dl = torch.tensor(data)  # tensor from list
data_array = np.array(data)  # numpy nd array from list
dn = torch.from_numpy(data_array)
print(f"from list:\n{dl}")
print()
print(f"from numpy array (not nd):\n{dn}")

# %%

d_ones = torch.ones_like(dl)
print(f"d_ones {d_ones}")

d_onesn = torch.ones(2, 2)
print(f"d_onesn {d_onesn}")

d_rand = torch.rand_like(dl, dtype=torch.float)  # sampled from standard normal
print(f"Random Tensor \n  {d_rand} \n")

# %% [markdown] Using Shape to create Tensors
#
# %%

shape = (2, 2, 3)
trand = torch.rand(shape)
tones = torch.ones(shape)
tzeroes = torch.zeros(shape)

for i, j in zip(("rand", "ones", "zeros"), (trand, tones, tzeroes)):
    print(f"Tensor using {i}: \n {j} \n")

# %% [markdown] Tensor attributes
# Attributes describe shape, datatype and the device a tensor is stored on.
#

# %%

ta = torch.randint(low=5, high=100, size=(2, 3))
attributes = {
    "shape": ta.shape,
    "datatype": ta.dtype,
    "device": ta.device,
}

for key, value in attributes.items():
    print()
    print(f"{key} is {value}")

# %% [markdown] Tensor operations
# See official docs for all operations there are.
#
# %%

tensor = torch.rand(2, 3)
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
    print(f"device tensor is stored on is: \n {tensor.device} \n")

vv = torch.from_numpy(np.arange(0, 9).reshape(3, 3))
print(f"vv \n {vv} \n")

# get slice that contains 4,5,7,8
slv = vv[1:, 1:]  # omitting k which is by default 1
slv2 = vv[1, 1]  # no slice but single element access
print(f"slice \n {slv} \n")
print(f"slice2 \n {slv2} \n")

tensor = torch.zeros(4, 4)
tensor[:, 1] = 1  # column 1 should be 1
print(f"tensor \n {tensor} \n")

# concat tensors column wise
t4 = torch.cat([tensor, tensor, tensor, tensor], dim=1)
print(
    f"Four times column wise concatenated tensor: \n {t4} \n\
Shape: \n {t4.shape} \n"
)


# concat tensors row wise
t4 = torch.cat([tensor, tensor, tensor, tensor], dim=0)
print(
    f"Four times row-wise concatenated tensor: \n {t4} \n\
Shape: \n {t4.shape} \n"
)

# multiplying tensors
tensor = torch.from_numpy(np.arange(0, 9).reshape(3, 3))
tensorm = tensor.mul(tensor)
tensorm2 = tensor * tensor
tensormm = tensor.matmul(tensor)
tensormm2 = tensor @ tensor

for i, j in zip(
    (
        "base tensor",
        "torch element wise multiplication tensorm",
        "python default element wise multiplication tensorm",
        "torch matrix multiplication tensormm",
        "python default matrix multiplication tensormm",
    ),
    (tensor, tensorm, tensorm2, tensormm, tensormm2),
):
    print(f"{i} \n {j} \n")

# %% [markdown]
# Inplace operations have suffix "_".
# 
# 
#
# %%

print(f'Tensor before \n {tensor} \n')
tensor.add_(5)
print(f'Tensor after \n {tensor} \n')

# %% [markdown] Bridge to numpy
# Bridge to numpy
#
# %%

# tensor to numpy
print(f'tensor before \n {tensor} \n')
nn = tensor.numpy()
print(f'tensor after to numpy \n {nn} \n')

# changes in tensor 'tensor' are reflected in numpy array
print(f'numpy version before \n {nn} \n')
print(f'tensor before in-place operation \n {tensor} \n')
print(f'tensor after in-place operation \n {tensor.add_(5)} \n')
print(f'numpy array after in-place operation \n {nn} \n')

# changes in a tensor, originating from numpy array are reflected in the
# original numpy array.
rng = np.random.default_rng()
arr = np.array(rng.standard_exponential(size=12)).reshape(3,4)
tnp = torch.from_numpy(arr)
print(f'numpy array before in-place operation \n {arr} \n')
print(f'tensor from numpy array before in-place operation \n {tnp} \n')
tnp.sub_(4)
print(f'tensor from numpy array after in-place operation on itself \n {tnp} \n')
print(f'numpy array after in-place operation on tensor \n {arr} \n')

# %% [markdown]
# End of tutorial, testing some things
# 
# 
#
# %%

torch.cos(tnp)
