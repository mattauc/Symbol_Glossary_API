from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision.transforms as transforms
from symbol_dataset import SymbolDataset
from pandas.core.common import flatten
import glob
import random
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DATA_PATH = "data/train"
TEST_DATA_PATH = "data/test"

train_image_paths = []
classes = []

# Load train dataset and class name
for data_path in glob.glob(TRAIN_DATA_PATH + '/*'):
    class_name = os.path.basename(data_path)
    if class_name not in classes:
        classes.append(class_name)
    train_image_paths.append(glob.glob(data_path + '/*'))
    
train_image_paths = list(flatten(train_image_paths))
random.shuffle(train_image_paths)

# Load train dataset
transforms = transforms.Compose([
        transforms.Resize((40, 40)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

train_dataset = SymbolDataset(image_paths=train_image_paths, classes=classes, transform=transforms)

# Create a data loader for the train dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0) # Look into batch size and workers

# Load test dataset
test_image_paths = []
for data_path in glob.glob(TEST_DATA_PATH + '/*'):
    test_image_paths.append(glob.glob(data_path + '/*'))
    
test_image_paths = list(flatten(test_image_paths))
test_dataset = SymbolDataset(image_paths=test_image_paths, classes=classes)

# Create a data loader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)


class OCRNet(nn.Module):
    def __init__(self, class_number):
        super(OCRNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 40 * 40, class_number)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = OCRNet(len(classes))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

# Training and evaluation loop
def run():
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss}')
        
        accuracy = evaluate(model, test_loader, device)
        print(f'Accuracy of the model on the test images: {accuracy}%')

if __name__ == '__main__':
    run()
