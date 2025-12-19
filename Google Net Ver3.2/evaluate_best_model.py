"""
Evaluate Best Model - GoogLeNet v3.2
Load the best saved model and evaluate on training set
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters - VERSION 3.2
BATCH_SIZE = 64
NUM_CLASSES = 4
IMG_SIZE = 224

# Data preprocessing - Same as training
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== GOOGLENET MODEL ====================

class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(True),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(True),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=4):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ==================== EVALUATION FUNCTION ====================

def evaluate_on_train_set(model, train_loader, criterion):
    """Evaluate the model on training set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    print("\nEvaluating on training set...")
    with torch.no_grad():
        pbar = tqdm(train_loader, desc='Evaluating')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


# ==================== MAIN ====================

print("\n" + "="*60)
print("EVALUATING BEST MODEL - GoogLeNet v3.2")
print("="*60)

# Load datasets
print("\nLoading training dataset...")
full_train_dataset = datasets.ImageFolder('Train', transform=train_transform)

train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Training set size: {len(train_dataset)}")
print(f"Classes: {full_train_dataset.classes}")

# Create model
print("\nCreating model...")
model = GoogLeNet(num_classes=NUM_CLASSES).to(device)

# Load best model
model_path = 'best_GoogLeNet_v3.2.pth'
print(f"\nLoading best model from {model_path}...")

try:
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Model loaded successfully!")
    print(f"  Saved from epoch: {checkpoint['epoch']+1}")
    print(f"  Validation accuracy: {checkpoint['accuracy']:.2f}%")

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate on training set
    print("\n" + "="*60)
    print("EVALUATING ON TRAINING SET")
    print("="*60)

    train_loss, train_acc = evaluate_on_train_set(model, train_loader, criterion)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Training Set Loss: {train_loss:.4f}")
    print(f"Training Set Accuracy: {train_acc:.2f}%")
    print("="*60)

    # Save results
    results = {
        'model_path': model_path,
        'saved_epoch': int(checkpoint['epoch'] + 1),
        'saved_val_accuracy': float(checkpoint['accuracy']),
        'training_loss': float(train_loss),
        'training_accuracy': float(train_acc)
    }

    with open('best_model_training_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n[SAVED] Results saved to best_model_training_metrics.json")

except FileNotFoundError:
    print(f"Error: Could not find model file '{model_path}'")
    print("Please make sure you have trained the model first using train_googlenet_ver3.2.py")
except Exception as e:
    print(f"Error loading model: {e}")
