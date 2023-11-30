import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [45000, 5000])
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
assert len(train_dataset) == 45000
assert len(val_dataset) == 5000

# Convert datasets to DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Visualize one image from CIFAR
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(train_loader)
# images, labels = dataiter.next() # doesn't work for python3
images, labels = next(dataiter) # works for python3
imshow(torchvision.utils.make_grid(images))

class SimpleCNN(nn.Module):
    def __init__(self):
    ###### Your code starts here. ######
    # layer definition
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 1024) #??? or 8 or 7
        self.fc2 = nn.Linear(1024, 10)
        self.dropout = nn.Dropout(0.5)
    ###### Your codes end here. ######

    def forward(self, x):
    ###### Your code starts here. ######
    # forwarding definition
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.log_softmax(self.fc2(x), dim=1)
        return x
    ###### Your codes end here. ######
    

# Training loop
def train_loop(model, train_loader, val_loader, criterion, optimizer):
    # n_epochs = 20
    n_epochs = 1    # for debug
    for epoch in range(n_epochs):
        model.train()
        for inputs, labels in train_loader:
            ###### Your code starts here. ######
            # do forward-backward propogation, and update the model
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            ###### Your code starts here. ######

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                ###### Your code starts here. ######
                # compute the validation loss and accuracy
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                ###### Your code starts here. ######
                correct += (predicted == labels.squeeze()).sum().item()

        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {val_loss/len(val_loader):.4f}, Accuracy: {correct/total:.4f}')

        # Evaluation
def test_loop(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            ###### Your code starts here. ######
            # compute the validation loss and accuracy
            # should be the same as the code for validation set
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            ###### Your code starts here. ######
            correct += (predicted == labels.squeeze()).sum().item()

    print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {correct/total:.4f}')

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

train_loop(model, train_loader, val_loader, criterion, optimizer)
test_loop(model, test_loader, criterion)