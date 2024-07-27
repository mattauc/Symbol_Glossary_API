import torch
from torch.utils.data import DataLoader
from .dataset import train_dataset, test_dataset, classes
from .model import OCRNet
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

model = OCRNet(len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# def show_image(image, label, class_names):
#     """Display a single image with its label."""
    
#     # Check the tensor dimensions
#     if image.dim() == 4:
#         image = image.squeeze(0)  # Remove batch dimension if present
#     elif image.dim() != 3:
#         raise ValueError("Unexpected image tensor dimension")

#     # Convert from Tensor format to numpy format (C, H, W) to (H, W, C)
#     image = image.numpy().transpose((1, 2, 0))  # Convert (C, H, W) to (H, W, C)

#     plt.figure(figsize=(6, 6))  # Adjust the size as needed
#     plt.imshow(image)
#     plt.title(class_names[label.item()])  # Convert label tensor to scalar
#     plt.axis('off')
#     plt.show()

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.early_stop_counter = 0

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
            return False

        if val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.early_stop_counter = 0
            return False
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= self.patience:
                return True

        return False

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

        # for m in range(len(images)):
        #     show_image(images[m].cpu(), labels[m].cpu(), classes)
    return running_loss / len(train_loader)

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()  # Assuming this is your criterion

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total, running_loss / len(test_loader)

def run():

    
    num_epochs = 20
    patience = 5
    early_stopping = EarlyStopping(patience=patience, min_delta=0.01)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss}')

        accuracy, val_loss = evaluate(model, test_loader, device)
        print(f'Accuracy of the model on the test images: {accuracy}%')
        print(f'Validation Loss: {val_loss}')

        # Early stopping check
        if early_stopping(val_loss):
            print("Early stopping triggered")
            break

     # Save the model weights
    torch.save(model.state_dict(), 'model_weights.pth')    

if __name__ == '__main__':
    run()