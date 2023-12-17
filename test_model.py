import torch

from model import VGG19

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VGG19(in_channels=3).to(device)


total_params = sum(p.numel() for p in model.parameters())
print(f"[INFO]: {total_params:,} total parameters.")
image_tensor = torch.randn(64, 3, 224, 224).to(device)  # a single image batch
outputs = model(image_tensor)
print(outputs.shape)
