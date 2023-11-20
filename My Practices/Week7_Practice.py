import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class BinaryCNN(nn.Module):
    def __init__(self):
        super(BinaryCNN, self).__init__()
        # Create a convolution layer with 1 input channels, 16 filters, 3x3 kernel size
        self.conv1 = nn.Conv2d(1, 16, 3)
        # Create a max pooling layer with 3x3 kernel
        self.pool1 = nn.MaxPool2d(3, 3)
        # Create a convolution layer with 32 input channels, 64 filters, 3x3 kernel size
        self.conv2 = nn.Conv2d(16, 64, 3)
        # Create a max pooling layer with 3x3 kernel
        self.pool2 = nn.MaxPool2d(2, 2)

        # Now do the Fully connected layers
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

        # Add a 20% drop out
        self.dropout = nn.Dropout(0.2)

    # Complete the forwarding function
    def forward(self, x):
        # Convolution and Activation (Layer 1)
        x = nn.relu(self.conv1(x))
        x = self.pool1(x)
        # Convolution and Activation (Layer 2)
        x = nn.relu(self.conv2(x))
        x = self.pool2(x)
        # Flattening
        x = x.view(-1, 64 * 3 * 3)
        # Fully Connected Layers with Dropout
        x = nn.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.relu(self.fc2(x))
        x = self.dropout(x)
        # Final Output Layer
        x = self.fc3(x)
        return x

def visualize_batch(images, labels, n=5):
    plt.figure(figsize=(10, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        ax.set_title(f'Label: {labels[i].item()}')
        plt.axis('off')
    plt.show()

def filter_01(labels, dataset):
    indices = (labels == 0) | (labels == 1)
    labels = labels[indices]
    dataset = dataset[indices]
    return labels, dataset

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loadewr = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

train_labels = train_dataset.targets
train_data = train_dataset.data
test_labels = test_dataset.targets
test_data = test_dataset.data

train_labels, train_data = filter_01(train_labels, train_data)
test_labels, test_data = filter_01(test_labels, test_data)

print("Filtered training set:")
visualize_batch(train_data, train_labels, n=5)
print("Filtered testing set:")
visualize_batch(test_data, test_labels, n=5)