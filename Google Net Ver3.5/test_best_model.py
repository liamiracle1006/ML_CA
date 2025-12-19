"""
Test Best GoogLeNet Model (v3.5) on Test Dataset
Complete evaluation with detailed metrics
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

# Test transform (same as training)
test_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load test dataset
print("\nLoading test dataset...")
test_dataset = datasets.ImageFolder('Test', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Test set size: {len(test_dataset)}")
print(f"Classes: {test_dataset.classes}")
print(f"Class to index mapping: {test_dataset.class_to_idx}")

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


# ==================== LOAD MODEL ====================

print("\n" + "="*60)
print("LOADING BEST MODEL")
print("="*60)

model = GoogLeNet(num_classes=NUM_CLASSES).to(device)
checkpoint = torch.load('best_GoogLeNet_v3.5.pth', map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded from: best_GoogLeNet_v3.5.pth")
print(f"Trained for: {checkpoint['epoch']+1} epochs")
print(f"Validation accuracy: {checkpoint['accuracy']:.2f}%")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ==================== EVALUATE ON TEST SET ====================

print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)

all_predictions = []
all_labels = []
all_probabilities = []

correct = 0
total = 0

with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {total}/{len(test_dataset)} samples...")

test_accuracy = 100. * correct / total

print("\n" + "="*60)
print("TEST RESULTS")
print("="*60)
print(f"\nOverall Test Accuracy: {test_accuracy:.4f}%")
print(f"Correct predictions: {correct}/{total}")

# ==================== DETAILED METRICS ====================

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)

# Classification report
class_names = test_dataset.classes
report = classification_report(all_labels, all_predictions,
                               target_names=class_names,
                               digits=4)
print(report)

# Get classification report as dict for saving
report_dict = classification_report(all_labels, all_predictions,
                                    target_names=class_names,
                                    output_dict=True)

# ==================== CONFUSION MATRIX ====================

print("\n" + "="*60)
print("CONFUSION MATRIX")
print("="*60)

cm = confusion_matrix(all_labels, all_predictions)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Print confusion matrix in text format
print("\nConfusion Matrix (counts):")
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
print("True\\Pred", end="")
for class_name in class_names:
    print(f"\t{class_name[:8]}", end="")
print()
for i, class_name in enumerate(class_names):
    print(f"{class_name[:8]}", end="")
    for j in range(len(class_names)):
        print(f"\t{cm_percent[i, j]:.2f}%", end="")
    print()

# Per-class accuracy
print("\n" + "="*60)
print("PER-CLASS ACCURACY")
print("="*60)
for i, class_name in enumerate(class_names):
    class_acc = cm_percent[i, i]
    class_total = cm[i].sum()
    class_correct = cm[i, i]
    print(f"{class_name}: {class_acc:.2f}% ({class_correct}/{class_total})")

# ==================== VISUALIZE CONFUSION MATRIX ====================

plt.figure(figsize=(12, 10))

annotations = np.empty_like(cm).astype(str)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}, linewidths=1, linecolor='gray')

plt.title('GoogLeNet v3.5 - Test Set Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.tight_layout()

plt.savefig('GoogLeNet_v3.5_test_confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"\n[SAVED] GoogLeNet_v3.5_test_confusion_matrix.png")
plt.close()

# ==================== SAVE RESULTS ====================

results = {
    'model': 'GoogLeNet v3.5',
    'checkpoint_file': 'best_GoogLeNet_v3.5.pth',
    'test_accuracy': test_accuracy,
    'total_samples': total,
    'correct_predictions': correct,
    'test_dataset_size': len(test_dataset),
    'classes': class_names,
    'per_class_accuracy': {
        class_names[i]: {
            'accuracy': float(cm_percent[i, i]),
            'correct': int(cm[i, i]),
            'total': int(cm[i].sum())
        } for i in range(len(class_names))
    },
    'confusion_matrix': cm.tolist(),
    'confusion_matrix_percent': cm_percent.tolist(),
    'classification_report': report_dict
}

with open('test_results_v3.5.json', 'w') as f:
    json.dump(results, f, indent=2)
print("[SAVED] test_results_v3.5.json")

print("\n" + "="*60)
print("TEST COMPLETED")
print("="*60)
print(f"\nFinal Test Accuracy: {test_accuracy:.4f}%")
print("\nAll results saved:")
print("  - GoogLeNet_v3.5_test_confusion_matrix.png")
print("  - test_results_v3.5.json")
print("="*60)
