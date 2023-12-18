import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from model import VGG19

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 30
batch_size = 32
learning_rate = 0.001
mtm = 0.9
wd = 0.0005

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5)),
    ]
)

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

valid_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False
)

model = VGG19(in_channels=1, num_classes=10).to(device)

total_params = sum(p.numel() for p in model.parameters())
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

criterion = nn.CrossEntropyLoss()
"""
optimizer = optim.SGD(
    model.parameters(), lr=learning_rate, momentum=mtm, weight_decay=wd
)
"""
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"[INFO]: Computation device: {device}")
print(f"[INFO]: {total_params:,} total parameters.")
print(f"[INFO]: {total_trainable_params:,} trainable parameters.")


def training(model, dataloader, optimizer, criterion):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0

    print("training")

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, labels)
        train_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)

        train_running_correct += (preds == labels).sum().item()
        loss.backward()
        optimizer.step()

    epoch_loss = train_running_loss / counter
    epoch_acc = 100.0 * (train_running_correct / len(dataloader.dataset))
    return epoch_loss, epoch_acc


def validating(model, dataloader, criterion):
    model.eval()
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))

    print("validing")

    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            output = model(image)
            loss = criterion(output, labels)
            valid_running_loss += loss.item()
            _, preds = torch.max(output.data, 1)
            valid_running_correct += (preds == labels).sum().item()

            correct = (preds == labels).squeeze()

            for i in range(len(preds)):
                label = labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    epoch_loss = valid_running_loss / counter
    epoch_acc = 100.0 * (valid_running_correct / len(dataloader.dataset))
    print("\n")
    for i in range(10):
        print(f"Accuracy of digit {i}: {100*class_correct[i]/class_total[i]}")

    return epoch_loss, epoch_acc


train_loss, valid_loss = [], []
train_acc, valid_acc = [], []


for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")

    train_epoch_loss, train_epoch_acc = training(
        model, train_dataloader, optimizer, criterion
    )
    valid_epoch_loss, valid_epoch_acc = validating(model, valid_dataloader, criterion)

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)

    print("\n")
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(
        f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}"
    )
    print("-" * 50)


torch.save(model.state_dict(), "./model.pth")

plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.plot(train_loss, label="Training Loss")
plt.plot(valid_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")

plt.subplot(2, 1, 2)
plt.plot(train_acc, label="Training Accuracy")
plt.plot(valid_acc, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Validation Accuracy")

# Save the figure as an image
plt.savefig("./training_validation_plots.png")
