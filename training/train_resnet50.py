import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.modules.loss import BCEWithLogitsLoss
from tqdm import tqdm

from model import ResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 30
batch_size = 32
learning_rate = 0.001

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

train_dataset = torchvision.datasets.ImageFolder(
    root="./dataset/training_dataset/", transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

valid_dataset = torchvision.datasets.ImageFolder(
    root="./dataset/validation_dataset/", transform=transform
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False
)

# model = ResNet(blocks=[3, 4, 6, 3], num_classes=10).to(device)
model = torchvision.models.resnet50(pretrained=True)

for params in model.parameters():
    params.requires_grad_ = False

nr_filters = model.fc.in_features
model.fc = nn.Linear(nr_filters, 1)

model = model.to(device)
optimizer = torch.optim.Adam(model.fc.parameters())


criterion = BCEWithLogitsLoss()

losses = []
val_losses = []

epoch_train_losses = []
epoch_test_losses = []


total_params = sum(p.numel() for p in model.parameters())
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

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
        labels = labels.unsqueeze(1).float().to(device)

        optimizer.zero_grad()
        output = model(image)

        loss = criterion(output, labels)
        train_running_loss += loss.item()

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

            output = model(image)[:, 0]
            loss = criterion(output, labels.float())
            valid_running_loss += loss.item()

    epoch_loss = valid_running_loss / counter
    epoch_acc = 100.0 * (valid_running_correct / len(dataloader.dataset))
    return epoch_loss, epoch_acc


train_loss, valid_loss = [], []
train_acc, valid_acc = [], []


for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")

    train_epoch_loss, train_epoch_acc = training(
        model, train_loader, optimizer, criterion
    )
    valid_epoch_loss, valid_epoch_acc = validating(model, valid_loader, criterion)

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


torch.save(model.state_dict(), "./model_ResNet50.pth")
