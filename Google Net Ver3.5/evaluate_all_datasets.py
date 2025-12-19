"""
Evaluate Best GoogLeNet Model (v3.5) on All Datasets
Comprehensive evaluation on Training, Validation, and Test sets
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Parameters
BATCH_SIZE = 64
NUM_CLASSES = 4
IMG_SIZE = 224

# Data preprocessing (same as training)
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

# Load datasets
print("\nLoading datasets...")
full_train_dataset = datasets.ImageFolder('Train', transform=train_transform)
test_dataset = datasets.ImageFolder('Test', transform=test_transform)

# Split train into train and validation (same split as training)
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

# Use the same random seed to get the same split as training
torch.manual_seed(42)
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Training set: {len(train_dataset)} samples")
print(f"Validation set: {len(val_dataset)} samples")
print(f"Test set: {len(test_dataset)} samples")
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

def evaluate_dataset(model, data_loader, dataset_name, class_names):
    """Evaluate model on a dataset and return detailed metrics"""
    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []

    correct = 0
    total = 0

    print(f"\nEvaluating on {dataset_name}...")

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    accuracy = 100. * correct / total

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Get classification report
    report_dict = classification_report(all_labels, all_predictions,
                                        target_names=class_names,
                                        output_dict=True,
                                        zero_division=0)

    # Per-class accuracy
    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        per_class_acc[class_name] = {
            'accuracy': float(cm_percent[i, i]),
            'correct': int(cm[i, i]),
            'total': int(cm[i].sum())
        }

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'confusion_matrix': cm,
        'confusion_matrix_percent': cm_percent,
        'per_class_accuracy': per_class_acc,
        'classification_report': report_dict,
        'predictions': all_predictions,
        'labels': all_labels
    }


# ==================== LOAD MODEL ====================

print("\n" + "="*70)
print("LOADING BEST MODEL")
print("="*70)

model = GoogLeNet(num_classes=NUM_CLASSES).to(device)
checkpoint = torch.load('best_GoogLeNet_v3.5.pth', map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded from: best_GoogLeNet_v3.5.pth")
print(f"Trained for: {checkpoint['epoch']+1} epochs")
print(f"Best validation accuracy during training: {checkpoint['accuracy']:.2f}%")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ==================== EVALUATE ON ALL DATASETS ====================

class_names = full_train_dataset.classes

print("\n" + "="*70)
print("COMPREHENSIVE EVALUATION ON ALL DATASETS")
print("="*70)

# Evaluate on training set
train_results = evaluate_dataset(model, train_loader, "Training Set", class_names)

# Evaluate on validation set
val_results = evaluate_dataset(model, val_loader, "Validation Set", class_names)

# Evaluate on test set
test_results = evaluate_dataset(model, test_loader, "Test Set", class_names)

# ==================== PRINT RESULTS ====================

print("\n" + "="*70)
print("OVERALL ACCURACY SUMMARY")
print("="*70)
print(f"Training Set:   {train_results['accuracy']:.4f}% ({train_results['correct']}/{train_results['total']})")
print(f"Validation Set: {val_results['accuracy']:.4f}% ({val_results['correct']}/{val_results['total']})")
print(f"Test Set:       {test_results['accuracy']:.4f}% ({test_results['correct']}/{test_results['total']})")

# Print detailed results for each dataset
for dataset_name, results in [("Training", train_results), ("Validation", val_results), ("Test", test_results)]:
    print("\n" + "="*70)
    print(f"{dataset_name.upper()} SET - DETAILED RESULTS")
    print("="*70)

    print(f"\nOverall Accuracy: {results['accuracy']:.4f}%")
    print(f"Correct: {results['correct']}/{results['total']}")

    print("\nPer-Class Accuracy:")
    for class_name in class_names:
        acc_info = results['per_class_accuracy'][class_name]
        print(f"  {class_name:10s}: {acc_info['accuracy']:6.2f}% ({acc_info['correct']}/{acc_info['total']})")

    print("\nConfusion Matrix (counts):")
    cm = results['confusion_matrix']
    print("True\\Pred", end="")
    for class_name in class_names:
        print(f"\t{class_name[:8]}", end="")
    print()
    for i, class_name in enumerate(class_names):
        print(f"{class_name[:8]}", end="")
        for j in range(len(class_names)):
            print(f"\t{cm[i, j]}", end="")
        print()

    print("\nConfusion Matrix (percentages):")
    cm_percent = results['confusion_matrix_percent']
    print("True\\Pred", end="")
    for class_name in class_names:
        print(f"\t{class_name[:8]}", end="")
    print()
    for i, class_name in enumerate(class_names):
        print(f"{class_name[:8]}", end="")
        for j in range(len(class_names)):
            print(f"\t{cm_percent[i, j]:.2f}%", end="")
        print()

    print("\nClassification Report:")
    report = results['classification_report']
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    print("-" * 60)
    for class_name in class_names:
        if class_name in report:
            print(f"{class_name:<12} {report[class_name]['precision']:<12.4f} "
                  f"{report[class_name]['recall']:<12.4f} {report[class_name]['f1-score']:<12.4f} "
                  f"{int(report[class_name]['support']):<12d}")
    print("-" * 60)
    print(f"{'Accuracy':<12} {'':<12} {'':<12} {report['accuracy']:<12.4f} {int(report['macro avg']['support']):<12d}")
    print(f"{'Macro Avg':<12} {report['macro avg']['precision']:<12.4f} "
          f"{report['macro avg']['recall']:<12.4f} {report['macro avg']['f1-score']:<12.4f} "
          f"{int(report['macro avg']['support']):<12d}")
    print(f"{'Weighted Avg':<12} {report['weighted avg']['precision']:<12.4f} "
          f"{report['weighted avg']['recall']:<12.4f} {report['weighted avg']['f1-score']:<12.4f} "
          f"{int(report['weighted avg']['support']):<12d}")

# ==================== VISUALIZE CONFUSION MATRICES ====================

print("\n" + "="*70)
print("GENERATING CONFUSION MATRIX VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, (dataset_name, results) in enumerate([("Training", train_results),
                                                ("Validation", val_results),
                                                ("Test", test_results)]):
    cm = results['confusion_matrix']
    cm_percent = results['confusion_matrix_percent']

    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, linewidths=1, linecolor='gray',
                ax=axes[idx])

    axes[idx].set_title(f'{dataset_name} Set\nAccuracy: {results["accuracy"]:.2f}%',
                        fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontsize=10, fontweight='bold')
    axes[idx].set_xlabel('Predicted Label', fontsize=10, fontweight='bold')

plt.suptitle('GoogLeNet v3.5 - Confusion Matrices Across All Datasets',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('GoogLeNet_v3.5_all_datasets_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("[SAVED] GoogLeNet_v3.5_all_datasets_confusion_matrices.png")
plt.close()

# ==================== SAVE COMPREHENSIVE RESULTS ====================

comprehensive_results = {
    'model': 'GoogLeNet v3.5',
    'checkpoint_file': 'best_GoogLeNet_v3.5.pth',
    'trained_epochs': checkpoint['epoch'] + 1,
    'total_parameters': total_params,
    'training_set': {
        'accuracy': train_results['accuracy'],
        'correct': train_results['correct'],
        'total': train_results['total'],
        'per_class_accuracy': train_results['per_class_accuracy'],
        'confusion_matrix': train_results['confusion_matrix'].tolist(),
        'confusion_matrix_percent': train_results['confusion_matrix_percent'].tolist(),
        'classification_report': train_results['classification_report']
    },
    'validation_set': {
        'accuracy': val_results['accuracy'],
        'correct': val_results['correct'],
        'total': val_results['total'],
        'per_class_accuracy': val_results['per_class_accuracy'],
        'confusion_matrix': val_results['confusion_matrix'].tolist(),
        'confusion_matrix_percent': val_results['confusion_matrix_percent'].tolist(),
        'classification_report': val_results['classification_report']
    },
    'test_set': {
        'accuracy': test_results['accuracy'],
        'correct': test_results['correct'],
        'total': test_results['total'],
        'per_class_accuracy': test_results['per_class_accuracy'],
        'confusion_matrix': test_results['confusion_matrix'].tolist(),
        'confusion_matrix_percent': test_results['confusion_matrix_percent'].tolist(),
        'classification_report': test_results['classification_report']
    },
    'classes': class_names
}

with open('comprehensive_evaluation_results_v3.5.json', 'w') as f:
    json.dump(comprehensive_results, f, indent=2)
print("[SAVED] comprehensive_evaluation_results_v3.5.json")

# ==================== FINAL SUMMARY ====================

print("\n" + "="*70)
print("EVALUATION COMPLETED")
print("="*70)

print("\nFinal Summary:")
print(f"  Training Set:   {train_results['accuracy']:.4f}%")
print(f"  Validation Set: {val_results['accuracy']:.4f}%")
print(f"  Test Set:       {test_results['accuracy']:.4f}%")

print("\nFiles Saved:")
print("  - GoogLeNet_v3.5_all_datasets_confusion_matrices.png")
print("  - comprehensive_evaluation_results_v3.5.json")

print("\n" + "="*70)
