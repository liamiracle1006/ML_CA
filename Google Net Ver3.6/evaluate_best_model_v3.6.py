"""
Evaluate Best GoogLeNet Model - Version 3.6
Load the best trained model and evaluate on all datasets (Train, Val, Test)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 128
NUM_CLASSES = 4
IMG_SIZE = 224

# Data preprocessing - same as training
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

test_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("\nLoading datasets...")
full_train_dataset = datasets.ImageFolder('Train', transform=test_transform)
test_dataset = datasets.ImageFolder('Test', transform=test_transform)

# Split train into train/val (same split as training)
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Training set: {len(train_dataset)}")
print(f"Validation set: {len(val_dataset)}")
print(f"Test set: {len(test_dataset)}")
print(f"Classes: {full_train_dataset.classes}")

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
        self.dropout = nn.Dropout(0.5)
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

def evaluate_model(model, data_loader, dataset_name):
    """Evaluate model on a given dataset"""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    print(f"\nEvaluating on {dataset_name} set...")

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100. * correct / total

    return accuracy, np.array(all_predictions), np.array(all_labels)


# ==================== MAIN ====================

print("\n" + "="*60)
print("EVALUATING BEST GOOGLENET MODEL - VERSION 3.6")
print("="*60)

# Load model
model = GoogLeNet(num_classes=NUM_CLASSES).to(device)
model_path = 'best_GoogLeNet_v3.6.pth'

print(f"\nLoading model from {model_path}...")
try:
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded successfully!")
    print(f"  - Saved at epoch: {checkpoint['epoch'] + 1}")
    print(f"  - Validation accuracy when saved: {checkpoint['accuracy']:.2f}%")
except FileNotFoundError:
    print(f"ERROR: Model file '{model_path}' not found!")
    print("Please make sure you have trained the model first.")
    exit(1)

# Evaluate on all datasets
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

# Training set
train_acc, train_pred, train_labels = evaluate_model(model, train_loader, "Training")
print(f"\nTraining Accuracy: {train_acc:.2f}%")

# Validation set
val_acc, val_pred, val_labels = evaluate_model(model, val_loader, "Validation")
print(f"Validation Accuracy: {val_acc:.2f}%")

# Test set
test_acc, test_pred, test_labels = evaluate_model(model, test_loader, "Test")
print(f"Test Accuracy: {test_acc:.2f}%")

# Per-class accuracy for each dataset
print("\n" + "="*60)
print("PER-CLASS ACCURACY")
print("="*60)

class_names = full_train_dataset.classes

for dataset_name, y_true, y_pred in [
    ("Training", train_labels, train_pred),
    ("Validation", val_labels, val_pred),
    ("Test", test_labels, test_pred)
]:
    print(f"\n{dataset_name} Set:")
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    for i, class_name in enumerate(class_names):
        class_acc = cm_percent[i, i]
        class_count = cm[i, i]
        total_count = cm[i].sum()
        print(f"  {class_name}: {class_acc:.2f}% ({class_count}/{total_count})")

# Detailed classification report for test set
print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT (Test Set)")
print("="*60)
print(classification_report(test_labels, test_pred, target_names=class_names, digits=4))

# Save results to JSON
results = {
    'model': 'GoogLeNet',
    'version': '3.6',
    'epoch_saved': int(checkpoint['epoch'] + 1),
    'accuracies': {
        'training': float(train_acc),
        'validation': float(val_acc),
        'test': float(test_acc)
    },
    'per_class_accuracy': {
        'training': {},
        'validation': {},
        'test': {}
    }
}

# Add per-class accuracies
for dataset_name, y_true, y_pred, key in [
    ("Training", train_labels, train_pred, 'training'),
    ("Validation", val_labels, val_pred, 'validation'),
    ("Test", test_labels, test_pred, 'test')
]:
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    for i, class_name in enumerate(class_names):
        results['per_class_accuracy'][key][class_name] = float(cm_percent[i, i])

output_file = 'evaluation_results_v3.6.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nGoogLeNet v3.6 (Epoch {checkpoint['epoch'] + 1}):")
print(f"  Training Accuracy:   {train_acc:.2f}%")
print(f"  Validation Accuracy: {val_acc:.2f}%")
print(f"  Test Accuracy:       {test_acc:.2f}%")

print(f"\n[SAVED] Results saved to {output_file}")
print("\n" + "="*60)
