import torch

x = torch.Tensor(5, 3)
# Randomize Tensor
y = torch.rand(5, 3)

# Add
print(x + y) # or
print(torch.add(x, y))

# Matrix Multiplication
a = torch.randn(2, 3)
b = torch.randn(3, 3)
print(torch.mm(a, b))
