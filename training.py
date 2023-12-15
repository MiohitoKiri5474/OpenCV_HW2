import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.models import vgg19_bn

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing and loading
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

train_dataset = MNIST(root="./drive/MyDrive/data", train=True, transform=transform, download=True)
train_dataset = [item for item in train_dataset if item[1] < 10]
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = MNIST(root="./drive/MyDrive/data", train=False, transform=transform, download=True)
train_dataset = [item for item in train_dataset if item[1] < 10]
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print("finish download")

# Model, loss function, and optimizer
model = vgg19_bn().to(device)

model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training and validation
train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []

print("finish loading module")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    valid_loss = 0.0
    correct_valid = 0
    total_valid = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_valid += labels.size(0)
            correct_valid += (predicted == labels).sum().item()

    valid_loss /= len(test_loader)
    valid_accuracy = correct_valid / total_valid

    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, "
        f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}"
    )

# Save model weights
torch.save(model.state_dict(), "./drive/MyDrive/vgg19_bn_mnist.pth")

# Plot training/validation loss and accuracy
plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.plot(train_losses, label="Training Loss")
plt.plot(valid_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")

plt.subplot(2, 1, 2)
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(valid_accuracies, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Validation Accuracy")

# Save the figure as an image
plt.savefig("./drive/MyDrive/training_validation_plots.png")

