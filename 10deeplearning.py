import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.autograd import Variable

from data import iris

# Create the module
class IrisNet(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(IrisNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden1_size)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden1_size, hidden2_size)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden2_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.act1(out)
        out = self.layer2(out)
        out = self.act2(out)
        out = self.layer3(out)
        return out


# Create a model instance
model = IrisNet(4, 100, 50, 3)
print(model)

# Create the DataLoader
batch_size = 60
iris_data_file = 'data/iris.data.txt'

train_ds, test_ds = iris.get_datasets(iris_data_file)

print('# instances in training set: ', len(train_ds))
print('# instances in testing/validation set: ', len(test_ds))

train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True)
