import torch

a = torch.ones(5)
print(a) # tensor([1., 1., 1., 1., 1.])
b = a.numpy()
print(b) # [1. 1. 1. 1. 1.]

import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a) # [2. 2. 2. 2. 2.]
print(b) # tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
