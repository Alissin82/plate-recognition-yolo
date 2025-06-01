import os
from typing import TextIO

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as f
from tqdm import tqdm
import json

# ---------- Load CSV with Encoded Labels ----------
csv_path = "../dataset/ocr/csv/iranis_labels_encoded.csv"
df = pd.read_csv(csv_path)

# Create a label mapping dictionary if not already created
unique_labels = sorted(df['label'].unique())
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}

mapping_file_path = "../dataset/ocr/json/"
if not os.path.isdir(mapping_file_path):
    os.makedirs(mapping_file_path)

# Save mapping for later use (e.g. during prediction)
with open(f"{mapping_file_path}index_to_label.json", "w") as json_file:
    json.dump(index_to_label, json_file)

# ---------- Custom PyTorch Dataset ----------
class PersianCharDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]        # image_path
        label = int(self.data.iloc[idx, 2])      # encoded label
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# ---------- Image Preprocessing Transforms ----------
transform = transforms.Compose([
    transforms.Resize((64, 64)),                      # Resize all images to 64x64
    transforms.ToTensor(),                            # Convert image to PyTorch Tensor
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)    # Normalize pixel values (for RGB channels)
])

# ---------- Initialize Dataset ----------
dataset = PersianCharDataset(csv_file=csv_path, transform=transform)

# ---------- Split into Training and Validation ----------
train_size = int(0.8 * len(dataset))  # 80% train, 20% validation
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ---------- Define Simple CNN Architecture ----------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)  # Input: (3x64x64) -> (32x62x62)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1) # -> (64x29x29)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)                 # After pooling: (64x14x14)
        self.fc2 = nn.Linear(128, num_classes)                  # Final classification layer

    def forward(self, x):
        x = f.relu(self.conv1(x))      # Apply ReLU activation
        x = f.max_pool2d(x, 2)         # Pooling -> (32x31x31)
        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x, 2)         # -> (64x14x14)
        x = torch.flatten(x, 1)        # Flatten for FC layer
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------- Set up Training ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(label_to_index)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ---------- Training Loop ----------
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    # Training loop with progress bar
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # ---------- Validation after each epoch ----------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}, Validation Accuracy = {accuracy:.2f}%")

# ---------- Save Trained Model ----------
model_path = "../dataset/ocr/model/"
if not os.path.isdir(model_path):
    os.makedirs(model_path)

torch.save(model.state_dict(), f"{model_path}persian_char_model.pth")
print(f"Model saved to {model_path}persian_char_model.pth")