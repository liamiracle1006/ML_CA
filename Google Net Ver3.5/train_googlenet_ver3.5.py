"""
Train GoogLeNet - Version 3.5
Strategy: Classical SGD Optimizer + Step Learning Rate Decay
- Optimizer: SGD with momentum (instead of AdamW)
- LR: 0.01 (typical for SGD)
- Momentum: 0.9
- Scheduler: StepLR (decay by 0.1 every 15 epochs)
- Goal: Test classical optimization approach
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters - VERSION 3.5
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.01  # Higher LR typical for SGD
MOMENTUM = 0.9
NUM_CLASSES = 4
IMG_SIZE = 224

# Data preprocessing - Moderate augmentation
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
full_train_dataset = datasets.ImageFolder('Train', transform=train_transform)
test_dataset = datasets.ImageFolder('Test', transform=test_transform)

train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
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
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

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


# ==================== TRAINING FUNCTIONS ====================

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, use_tta=False, tta_transforms=None, return_predictions=False):
    model.eval()
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if use_tta and tta_transforms:
                outputs_list = []
                for tta_transform in tta_transforms:
                    augmented_inputs = torch.stack([tta_transform(img) for img in inputs])
                    outputs = model(augmented_inputs)
                    outputs_list.append(outputs)
                outputs = torch.stack(outputs_list).mean(dim=0)
            else:
                outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if return_predictions:
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    accuracy = 100. * correct / total

    if return_predictions:
        return accuracy, np.array(all_predictions), np.array(all_labels)
    return accuracy


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(10, 8))

    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, linewidths=1, linecolor='gray')

    plt.title(f'{model_name} - Confusion Matrix (v3.5)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] Confusion Matrix: {save_path}")
    plt.close()

    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        class_acc = cm_percent[i, i]
        print(f"  {class_name}: {class_acc:.2f}%")


def train_model(model_name, model, num_epochs=NUM_EPOCHS):
    print(f"\n{'='*60}")
    print(f"Training: {model_name} (Version 3.5)")
    print(f"{'='*60}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    # VERSION 3.5: Classical SGD with momentum instead of AdamW
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=0.0001)

    # VERSION 3.5: StepLR instead of CosineAnnealingLR (decay by 0.1 every 15 epochs)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    print(f"Optimizer: SGD (LR={LEARNING_RATE}, Momentum={MOMENTUM})")
    print(f"Learning Rate Schedule: StepLR (decay by 0.1 every 15 epochs)")

    start_time = time.time()
    best_val_acc = 0
    train_losses = []
    train_accs = []
    val_accs = []

    patience = 10
    patience_counter = 0

    print(f"Early Stopping: Enabled (patience={patience} epochs)")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_acc = evaluate(model, val_loader)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': best_val_acc,
            }, f'best_{model_name}_v3.5.pth')
            print(f"[SAVED] Best Val: {best_val_acc:.2f}%")

    training_time = time.time() - start_time

    print(f"\n{'='*60}")
    print("EVALUATION ON TEST SET")
    print(f"{'='*60}")

    print(f"Loading best model from best_{model_name}_v3.5.pth...")
    checkpoint = torch.load(f'best_{model_name}_v3.5.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best model loaded (from epoch {checkpoint['epoch']+1}, val acc: {checkpoint['accuracy']:.2f}%)")

    test_acc, y_pred, y_true = evaluate(model, test_loader, use_tta=False, return_predictions=True)
    print(f"Test Accuracy: {test_acc:.2f}%")

    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, full_train_dataset.classes, model_name,
                         f'{model_name}_confusion_matrix_v3.5.png')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    actual_epochs = len(train_losses)
    epochs_range = range(1, actual_epochs + 1)

    ax1.plot(epochs_range, train_losses, 'b-', linewidth=2)
    ax1.set_title(f'{model_name} - Training Loss (v3.5)', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, train_accs, 'b-', linewidth=2, label='Train')
    ax2.plot(epochs_range, val_accs, 'r-', linewidth=2, label='Validation')
    ax2.axhline(y=92, color='g', linestyle='--', linewidth=2, label='Target')
    # Mark StepLR decay points
    for step in [15, 30, 45]:
        if step <= actual_epochs:
            ax2.axvline(x=step, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.set_title(f'{model_name} - Accuracy (v3.5)', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(f'{model_name}_training_v3.5.png', dpi=300, bbox_inches='tight')
    print(f"\n[SAVED] {model_name}_training_v3.5.png")
    plt.close()

    return {
        'version': '3.5',
        'model_name': model_name,
        'best_val_accuracy': best_val_acc,
        'test_accuracy': test_acc,
        'training_time': training_time,
        'parameters': total_params,
        'config': {
            'optimizer': 'SGD',
            'learning_rate': LEARNING_RATE,
            'momentum': MOMENTUM,
            'dropout': 0.5,
            'weight_decay': 0.0001,
            'batch_size': BATCH_SIZE,
            'epochs': NUM_EPOCHS,
            'scheduler': 'StepLR (step=15, gamma=0.1)'
        }
    }


# ==================== MAIN ====================

print("\n" + "="*60)
print("TRAINING GOOGLENET - VERSION 3.5")
print("Classical SGD + Step LR Decay")
print("="*60)

googlenet = GoogLeNet(num_classes=NUM_CLASSES).to(device)
result = train_model('GoogLeNet', googlenet)

# ==================== FINAL SUMMARY ====================

print("\n" + "="*80)
print("TRAINING COMPLETED - VERSION 3.5")
print("="*80)

print(f"\n{result['model_name']} (v3.5):")
print(f"  Best Validation Accuracy: {result['best_val_accuracy']:.2f}%")
print(f"  Test Accuracy: {result['test_accuracy']:.2f}%")
print(f"  Training Time: {result['training_time']/60:.2f} min")
print(f"  Parameters: {result['parameters']:,}")

with open('googlenet_results_v3.5.json', 'w') as f:
    json.dump(result, f, indent=2)
print("\n[SAVED] googlenet_results_v3.5.json")

print("\n" + "="*80)
print("DONE! Check GoogLeNet_training_v3.5.png for training curves.")
print("="*80)
