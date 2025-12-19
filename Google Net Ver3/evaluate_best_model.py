"""
Evaluate Best GoogLeNet Model on Test Data
Generates confusion matrix and per-class accuracy
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters (same as training)
NUM_CLASSES = 4
IMG_SIZE = 224
BATCH_SIZE = 64

# Test data preprocessing (same as training)
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

# ==================== GOOGLENET MODEL (same architecture as training) ====================

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
        self.dropout = nn.Dropout(0.4)
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


# ==================== LOAD BEST MODEL ====================

print("\n" + "="*60)
print("LOADING BEST MODEL")
print("="*60)

model = GoogLeNet(num_classes=NUM_CLASSES).to(device)

# Load the best model checkpoint
checkpoint = torch.load('best_GoogLeNet_optimized.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"[OK] Best model loaded from epoch {checkpoint['epoch']+1}")
print(f"[OK] Validation accuracy: {checkpoint['accuracy']:.2f}%")

# ==================== EVALUATE ON TEST SET ====================

print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)

model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# Calculate overall accuracy
overall_accuracy = 100. * np.sum(all_predictions == all_labels) / len(all_labels)
print(f"\nOverall Test Accuracy: {overall_accuracy:.2f}%")

# ==================== CONFUSION MATRIX ====================

print("\n" + "="*60)
print("GENERATING CONFUSION MATRIX")
print("="*60)

# Calculate confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
class_names = test_dataset.classes

# Create figure with larger size for better readability
plt.figure(figsize=(10, 8))

# Plot confusion matrix with annotations
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}, annot_kws={'size': 14})

plt.title('Confusion Matrix - GoogLeNet Test Set', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save confusion matrix
plt.savefig('confusion_matrix_test.png', dpi=300, bbox_inches='tight')
print("\n[SAVED] confusion_matrix_test.png")
plt.close()

# ==================== PER-CLASS ACCURACY ====================

print("\n" + "="*60)
print("PER-CLASS ACCURACY")
print("="*60)

# Calculate per-class metrics
class_accuracies = {}
class_totals = {}
class_correct = {}

for i, class_name in enumerate(class_names):
    class_mask = (all_labels == i)
    class_total = np.sum(class_mask)
    class_correct_count = np.sum((all_predictions == all_labels) & class_mask)
    class_accuracy = 100. * class_correct_count / class_total if class_total > 0 else 0

    class_accuracies[class_name] = class_accuracy
    class_totals[class_name] = int(class_total)
    class_correct[class_name] = int(class_correct_count)

    print(f"\n{class_name}:")
    print(f"  Correct: {class_correct_count}/{class_total}")
    print(f"  Accuracy: {class_accuracy:.2f}%")

# ==================== CLASSIFICATION REPORT ====================

print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT")
print("="*60)

# Generate classification report
report = classification_report(all_labels, all_predictions,
                               target_names=class_names, digits=4)
print("\n" + report)

# ==================== VISUALIZATION: PER-CLASS ACCURACY BAR CHART ====================

print("\n" + "="*60)
print("GENERATING PER-CLASS ACCURACY CHART")
print("="*60)

fig, ax = plt.subplots(figsize=(12, 6))

classes = list(class_accuracies.keys())
accuracies = list(class_accuracies.values())

# Create bar chart
bars = ax.bar(range(len(classes)), accuracies, color='steelblue', alpha=0.8, edgecolor='black')

# Add value labels on top of bars
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.2f}%\n({class_correct[classes[i]]}/{class_totals[classes[i]]})',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add overall accuracy line
ax.axhline(y=overall_accuracy, color='red', linestyle='--', linewidth=2,
           label=f'Overall Accuracy: {overall_accuracy:.2f}%')

ax.set_xlabel('Class', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Per-Class Accuracy on Test Set - GoogLeNet', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.set_ylim([0, 105])
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=11, loc='lower right')

plt.tight_layout()
plt.savefig('per_class_accuracy_test.png', dpi=300, bbox_inches='tight')
print("\n[SAVED] per_class_accuracy_test.png")
plt.close()

# ==================== SAVE RESULTS TO JSON ====================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

results = {
    'model_name': 'GoogLeNet_optimized',
    'checkpoint_epoch': int(checkpoint['epoch']) + 1,
    'validation_accuracy': float(checkpoint['accuracy']),
    'overall_test_accuracy': float(overall_accuracy),
    'per_class_accuracy': {class_name: float(acc) for class_name, acc in class_accuracies.items()},
    'per_class_correct': class_correct,
    'per_class_total': class_totals,
    'confusion_matrix': cm.tolist(),
    'class_names': class_names
}

with open('test_evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n[SAVED] test_evaluation_results.json")

# ==================== FINAL SUMMARY ====================

print("\n" + "="*80)
print("EVALUATION COMPLETED")
print("="*80)

print(f"\nModel: GoogLeNet (Optimized)")
print(f"Best Model from Epoch: {checkpoint['epoch']+1}")
print(f"Validation Accuracy: {checkpoint['accuracy']:.2f}%")
print(f"Test Accuracy: {overall_accuracy:.2f}%")

print(f"\nPer-Class Accuracy:")
for class_name, acc in class_accuracies.items():
    print(f"  {class_name}: {acc:.2f}% ({class_correct[class_name]}/{class_totals[class_name]})")

print(f"\nGenerated Files:")
print(f"  1. confusion_matrix_test.png - Confusion matrix heatmap")
print(f"  2. per_class_accuracy_test.png - Per-class accuracy bar chart")
print(f"  3. test_evaluation_results.json - Detailed results in JSON format")

print("\n" + "="*80)
print("DONE!")
print("="*80)
