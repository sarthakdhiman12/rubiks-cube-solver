import os
import json
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from model import RubiksFaceModel

DATA_DIR = "data/images"
LABEL_FILE = "data/dataset.json"
MODEL_PATH = "model.pth"
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001

class RubiksDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_name = os.path.join(self.root_dir, item['id'])
        image = Image.open(img_name).convert("RGB")
        labels = torch.tensor(item['labels'], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, labels

def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = RubiksDataset(LABEL_FILE, DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = RubiksFaceModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Starting training...")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.view(-1, 6), labels.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
