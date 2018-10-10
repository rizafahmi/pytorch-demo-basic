import torch

x = torch.Tensor(5, 3)
y = torch.rand(5, 3)

if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    x + y
