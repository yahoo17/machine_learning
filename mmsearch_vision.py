import torch
import torchvision.models

shape = (1,3,64,64)
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(shape)
labels = torch.rand(1,1000)

# forward

prediction = model(data)

loss = (prediction - labels).sum()

loss.backward()

optim = torch.optim.SGD(model.parameters(),lr=1e-2,momentum=0.9)

optim.step()