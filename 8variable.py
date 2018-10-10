import torch
from torch.autograd import Variable

x = Variable(torch.FloatTensor([11.2]), requires_grad=True)
y = 2 * x
print(x) # tensor([11.2000], requires_grad=True)
print(y) # tensor([22.4000], grad_fn=<MulBackward>)
print(x.data) # tensor([11.2000])
print(y.data) # tensor([22.4000])
print(x.grad_fn) # None
print(y.grad_fn) # <MulBackward object at 0x10ae58e48>
y.backward() # Calculates the gradients
print(x.grad) # tensor([2.])
print(y.grad) # None
