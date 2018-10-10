import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))
                                ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)
print(len(trainset.train_data))
plt.imshow(trainset.train_data[1])
print(trainset.train_labels[1]) # 9 = Truck image

trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                          shuffle=True, num_workers=2)

for i, data in enumerate(trainloader):
    data, labels = data

    print("Iteration ", i)
    print("==============")
    print("type(data): ", type(data))
    print("data.size(): ", data.size())
    print("==============")
    print("type(labels): ", type(labels))
    print("labels.size(): ", labels.size())

    # Model training happens here...

    break
