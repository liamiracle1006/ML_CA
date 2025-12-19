"""
Evaluate Best GoogLeNet Model - Version 3.3
Load the best model and evaluate on all datasets (Train, Val, Test)
Show comprehensive metrics including accuracy, precision, recall, F1-score
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters (same as training)
BATCH_SIZE = 64
NUM_CLASSES = 4
IMG_SIZE = 224

# Data preprocessing
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
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

# Split train into train and val (same split as training)
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
        self.dropout = nn.Dropout(0.6)
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


# ==================== EVALUATION FUNCTIONS ====================

def evaluate_detailed(model, data_loader, dataset_name, class_names):
    """Detailed evaluation with all metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []

    print(f"\nEvaluating on {dataset_name}...")

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate accuracy
    accuracy = 100. * np.sum(all_predictions == all_labels) / len(all_labels)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Classification report
    report = classification_report(all_labels, all_predictions,
                                   target_names=class_names,
                                   digits=4,
                                   output_dict=True)

    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels,
        'probs': all_probs,
        'confusion_matrix': cm,
        'classification_report': report
    }


def plot_confusion_matrix(cm, class_names, dataset_name, save_path):
    """Plot confusion matrix with counts and percentages"""
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(10, 8))

    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, linewidths=1, linecolor='gray')

    plt.title(f'Confusion Matrix - {dataset_name}', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
    plt.close()


def print_detailed_metrics(results, dataset_name, class_names):
    """Print detailed metrics for a dataset"""
    print(f"\n{'='*70}")
    print(f"{dataset_name} SET RESULTS")
    print(f"{'='*70}")
    print(f"Overall Accuracy: {results['accuracy']:.4f}%")
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)

    report = results['classification_report']
    for class_name in class_names:
        metrics = report[class_name]
        print(f"{class_name:<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1-score']:<12.4f} {int(metrics['support']):<10}")

    print("-" * 70)
    print(f"{'Macro Avg':<15} {report['macro avg']['precision']:<12.4f} "
          f"{report['macro avg']['recall']:<12.4f} {report['macro avg']['f1-score']:<12.4f}")
    print(f"{'Weighted Avg':<15} {report['weighted avg']['precision']:<12.4f} "
          f"{report['weighted avg']['recall']:<12.4f} {report['weighted avg']['f1-score']:<12.4f}")

    # Per-class accuracy from confusion matrix
    cm = results['confusion_matrix']
    print(f"\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        class_acc = 100. * cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  {class_name}: {class_acc:.2f}%")


# ==================== MAIN ====================

print("\n" + "="*70)
print("EVALUATING BEST GOOGLENET MODEL - VERSION 3.3")
print("="*70)

# Load model
print("\nLoading best model...")
model = GoogLeNet(num_classes=NUM_CLASSES).to(device)

try:
    checkpoint = torch.load('best_GoogLeNet_v3.3.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded successfully!")
    print(f"  Epoch: {checkpoint['epoch']+1}")
    print(f"  Validation Accuracy (during training): {checkpoint['accuracy']:.2f}%")
except FileNotFoundError:
    print("ERROR: best_GoogLeNet_v3.3.pth not found!")
    print("Please run train_googlenet_ver3.3.py first to train the model.")
    exit(1)

model.eval()

# Get class names
class_names = full_train_dataset.classes

# Evaluate on all datasets
train_results = evaluate_detailed(model, train_loader, "Training", class_names)
val_results = evaluate_detailed(model, val_loader, "Validation", class_names)
test_results = evaluate_detailed(model, test_loader, "Test", class_names)

# Print detailed metrics for all datasets
print_detailed_metrics(train_results, "TRAINING", class_names)
print_detailed_metrics(val_results, "VALIDATION", class_names)
print_detailed_metrics(test_results, "TEST", class_names)

# Plot confusion matrices
print("\n" + "="*70)
print("Generating confusion matrices...")
print("="*70)
plot_confusion_matrix(train_results['confusion_matrix'], class_names,
                     "Training Set", "confusion_matrix_train.png")
plot_confusion_matrix(val_results['confusion_matrix'], class_names,
                     "Validation Set", "confusion_matrix_val.png")
plot_confusion_matrix(test_results['confusion_matrix'], class_names,
                     "Test Set", "confusion_matrix_test.png")

# Summary comparison
print("\n" + "="*70)
print("SUMMARY - ALL DATASETS")
print("="*70)
print(f"{'Dataset':<15} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
print("-" * 70)

for name, results in [("Training", train_results),
                      ("Validation", val_results),
                      ("Test", test_results)]:
    report = results['classification_report']
    print(f"{name:<15} {results['accuracy']:<15.4f} "
          f"{report['weighted avg']['precision']:<15.4f} "
          f"{report['weighted avg']['recall']:<15.4f} "
          f"{report['weighted avg']['f1-score']:<15.4f}")

# Save results to JSON
evaluation_results = {
    'model': 'GoogLeNet',
    'version': '3.3',
    'training_set': {
        'accuracy': float(train_results['accuracy']),
        'precision': float(train_results['classification_report']['weighted avg']['precision']),
        'recall': float(train_results['classification_report']['weighted avg']['recall']),
        'f1_score': float(train_results['classification_report']['weighted avg']['f1-score']),
        'per_class': {class_name: {
            'precision': float(train_results['classification_report'][class_name]['precision']),
            'recall': float(train_results['classification_report'][class_name]['recall']),
            'f1_score': float(train_results['classification_report'][class_name]['f1-score']),
            'support': int(train_results['classification_report'][class_name]['support'])
        } for class_name in class_names}
    },
    'validation_set': {
        'accuracy': float(val_results['accuracy']),
        'precision': float(val_results['classification_report']['weighted avg']['precision']),
        'recall': float(val_results['classification_report']['weighted avg']['recall']),
        'f1_score': float(val_results['classification_report']['weighted avg']['f1-score']),
        'per_class': {class_name: {
            'precision': float(val_results['classification_report'][class_name]['precision']),
            'recall': float(val_results['classification_report'][class_name]['recall']),
            'f1_score': float(val_results['classification_report'][class_name]['f1-score']),
            'support': int(val_results['classification_report'][class_name]['support'])
        } for class_name in class_names}
    },
    'test_set': {
        'accuracy': float(test_results['accuracy']),
        'precision': float(test_results['classification_report']['weighted avg']['precision']),
        'recall': float(test_results['classification_report']['weighted avg']['recall']),
        'f1_score': float(test_results['classification_report']['weighted avg']['f1-score']),
        'per_class': {class_name: {
            'precision': float(test_results['classification_report'][class_name]['precision']),
            'recall': float(test_results['classification_report'][class_name]['recall']),
            'f1_score': float(test_results['classification_report'][class_name]['f1-score']),
            'support': int(test_results['classification_report'][class_name]['support'])
        } for class_name in class_names}
    }
}

with open('evaluation_results_all_datasets.json', 'w') as f:
    json.dump(evaluation_results, f, indent=2)
print("\n[SAVED] evaluation_results_all_datasets.json")

print("\n" + "="*70)
print("EVALUATION COMPLETED!")
print("="*70)
print("\nGenerated files:")
print("  - confusion_matrix_train.png")
print("  - confusion_matrix_val.png")
print("  - confusion_matrix_test.png")
print("  - evaluation_results_all_datasets.json")
print("="*70)
