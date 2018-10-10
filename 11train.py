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

# Model
net = IrisNet(4, 100, 50, 3)

# Loss Function
criterion = nn.CrossEntropyLoss()

# Optimizer
learning_rate = 0.001
optimizer = torch.optim.SGD(net.parameters(),
                            lr=learning_rate,
                            nesterov=True,
                            momentum=0.9,
                            dampening=0)

# Training iteration
num_epochs = 500

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

for epoch in range(num_epochs):
    train_correct = 0
    train_total = 0

    for i, (items, classes) in enumerate(train_loader):
        # Convert torch tensor to Variable
        items = Variable(items)
        classes = Variable(classes)

        net.train() # Training mode
        optimizer.zero_grad() # Reset gradients from past operation
        outputs = net(items) # Forward pass
        loss = criterion(outputs, classes) # Calculate the loss
        loss.backward() # Calculate the gradient
        optimizer.step() # Adjust weight/parameter based on gradients

        train_total += classes.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == classes.data).sum()

        print('Epoch %d/%d, Iteration %d/%d, Loss: %.4f'
              %(epoch+1, num_epochs, i+1, len(train_ds)//batch_size, loss.data[0]))

    net.eval() # Put the network into evaluation mode

    train_loss.append(loss.data[0])
    train_accuracy.append((100 * train_correct / train_total))

    # Record the testing loss
    test_items = torch.FloatTensor(test_ds.data.values[:, 0:4])
    test_classes = torch.LongTensor(test_ds.data.values[:, 4])
    outputs = net(Variable(test_items))
    loss = criterion(outputs, Variable(test_classes))
    test_loss.append(loss.data[0])

    # Record the testing accuracy
    _, predicted = torch.max(outputs.data, 1)
    total = test_classes.size(0)
    correct = (predicted == test_classes).sum()
    test_accuracy.append((100 * correct / total))
