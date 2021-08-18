import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] )

batch_size = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,transform=transform)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size= batch_size, shuffle=True, num_workers=0)

test_loader =torch.utils.data.DataLoader(test_set, batch_size= batch_size, shuffle=False, num_workers=0)


classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120,85)
        self.fc3 = nn.Linear(85,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x,1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
learning_rate = 0.0008
momentum = 0.9
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

for epoch in range(4):
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):

        inputs, label = data

        optimizer.zero_grad()

        output = net(inputs)

        loss = criterion(output, label)
        loss.backward()


        optimizer.step()

        running_loss += loss.item()

        if i%2000 == 1999:
            print('[%d,%5d] loss:%.3f' %(epoch+1, i + 1, running_loss/2000))
            running_loss = 0.0

print("finish training")


PATH = './cifar_net.pth'
torch.save(net.state_dict(),PATH)

dataiter = iter(test_loader)

images, labels = dataiter.next()

# imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net2 = Net()
net2.load_state_dict(torch.load(PATH))

outputs = net2(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net2(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted==labels).sum().item()


print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                   accuracy))