import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32

ResNet50_path = "./model/model_ResNet50.pth"
ResNet50_with_random_path = "./model/model_ResNet50_with_random.pth"

model1 = torchvision.models.resnet50().to(device)
nr_filters1 = model1.fc.in_features
model1.fc = nn.Linear(nr_filters1, 1)
state_dict1 = torch.load(ResNet50_path, map_location=torch.device(device))
model1.load_state_dict(state_dict1)
model1.to(device)
model1.eval()


model2 = torchvision.models.resnet50().to(device)
nr_filters2 = model2.fc.in_features
model2.fc = nn.Linear(nr_filters2, 1)
state_dict2 = torch.load(ResNet50_with_random_path, map_location=torch.device(device))
model2.load_state_dict(state_dict2)
model2.to(device)
model2.eval()

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

valid_dataset = ImageFolder(root="./dataset/validation_dataset/", transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


def calculation(model, loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            output = torch.sigmoid(model(images))

            lt = []
            for i in range(len(labels)):
                # print ( i )
                tmp = 0 if output[i, 0] < 0.5 else 1
                total += 1
                correct += (tmp == labels[i]).item()
                lt.append([tmp, labels[1]])

    accuracy = 100.0 * correct / total
    print("\tcorrect:\t", correct)
    print("\ttotal:\t", total)
    print("\taccuracy:\t", accuracy)
    return accuracy


acc1 = calculation(model1, valid_loader)
acc2 = calculation(model2, valid_loader)

models = ["ResNet50", "ResNet50 with random erasing"]
acc_rate = [acc1, acc2]

plt.bar(models, acc_rate, color=["Blue", "Blue"])

for i, value in enumerate(acc_rate):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center", va="bottom")

plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy Comparison")
plt.ylim(0, 100)
plt.savefig("ResNet50_comparison.png")
plt.show()
