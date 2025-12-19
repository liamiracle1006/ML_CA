import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, Activation, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import json
import time
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== Hyperparameters ====================
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLASSES = 4
IMG_SIZE = 224
TRAIN_DIR = 'Train'
TEST_DIR = 'Test'
VALIDATION_SPLIT = 0.2

print(f"Using device: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
print(f"TensorFlow version: {tf.__version__}")


# ==================== Data Loading (PyTorch-style) ====================

def load_datasets():
    """
    Load datasets with PyTorch-style ImageFolder approach
    Using ImageDataGenerator for data augmentation
    """
    print("\n" + "="*60)
    print("Loading datasets...")
    print("="*60)

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.6, 1.4],
        fill_mode='nearest',
        validation_split=VALIDATION_SPLIT
    )

    # Only normalization for validation and test
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VALIDATION_SPLIT
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = test_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    print(f"Training set: {train_generator.n}")
    print(f"Validation set: {val_generator.n}")
    print(f"Test set: {test_generator.n}")
    print(f"Classes: {list(train_generator.class_indices.keys())}")

    return train_generator, val_generator, test_generator


# ==================== Model Architecture ====================

def build_model(num_classes=NUM_CLASSES):
    """
    Build custom CNN model - FruitNet
    Simplified convolutional neural network for fruit classification
    """
    print("\n" + "="*60)
    print("Building FruitNet Model...")
    print("="*60)

    model = Sequential([
        # Conv Block 1: 32 filters
        Conv2D(32, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Conv Block 2: 64 filters
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Conv Block 3: 128 filters
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        # Conv Block 4: 256 filters
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        # Global pooling instead of Flatten
        GlobalAveragePooling2D(),

        # Fully connected layers
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),

        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),

        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    total_params = model.count_params()

    print(f"Model Architecture:")
    print(f"  - 4 Convolutional Blocks (32->64->128->256 filters)")
    print(f"  - Global Average Pooling")
    print(f"  - 2 Dense Layers (256->128)")
    print(f"  - Output Layer ({num_classes} classes)")
    print(f"\nTotal parameters: {total_params:,}")

    return model


def compile_model(model, learning_rate=LEARNING_RATE):
    """Compile model with optimizer and loss function"""
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"Model compiled with learning rate: {learning_rate}")


def get_class_weights(train_generator):
    """
    Compute class weights to handle class imbalance
    Give higher weights to classes with fewer samples
    """
    from sklearn.utils.class_weight import compute_class_weight

    class_indices = train_generator.class_indices
    classes = list(class_indices.keys())

    class_counts = {}
    for class_name in classes:
        class_dir = os.path.join(TRAIN_DIR, class_name)
        class_counts[class_name] = len([f for f in os.listdir(class_dir)
                                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    total_samples = sum(class_counts.values())
    num_classes = len(classes)

    # Calculate weights: higher weight for minority classes
    class_weight = {}
    for class_name, count in class_counts.items():
        class_idx = class_indices[class_name]
        weight = total_samples / (num_classes * count)
        class_weight[class_idx] = weight

    print("\nClass weights (to handle imbalance):")
    for class_name in classes:
        class_idx = class_indices[class_name]
        print(f"  {class_name}: {class_weight[class_idx]:.2f} (samples: {class_counts[class_name]})")

    return class_weight


# ==================== Training ====================

def train_model(model, train_generator, val_generator, epochs=NUM_EPOCHS):
    """
    Train model with PyTorch-style approach using Keras fit method
    """
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)

    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    class_weight = get_class_weights(train_generator)

    # Save best model & reduce learning rate on plateau
    callbacks = [
        ModelCheckpoint(
            f'models/best_FruitNet_{timestamp}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
    ]

    start_time = time.time()

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    training_time = time.time() - start_time

    final_model_path = f'models/final_FruitNet_{timestamp}.h5'
    model.save(final_model_path)

    history_dict = {
        'train_loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'train_accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'training_time': training_time,
        'epochs_trained': len(history.history['loss'])
    }

    history_path = f'logs/training_history_{timestamp}.json'
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)

    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    print(f"Best validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
    print(f"[SAVED] {final_model_path}")
    print(f"[SAVED] {history_path}")

    return history, timestamp


# ==================== Evaluation ====================

def evaluate_model(model, test_generator):
    """
    Evaluate model on test set
    """
    print(f"\n{'='*60}")
    print("EVALUATION ON TEST SET")
    print(f"{'='*60}")

    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")

    return test_accuracy


# ==================== Visualization ====================

def plot_training_curves(history):
    """
    Plot training curves (matching 图片处理&plot.py style)
    """
    train_losses = history.history['loss']
    train_accs = [x * 100 for x in history.history['accuracy']]
    val_accs = [x * 100 for x in history.history['val_accuracy']]

    actual_epochs = len(train_losses)
    epochs_range = range(1, actual_epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Training loss
    ax1.plot(epochs_range, train_losses, 'b-', linewidth=2)
    ax1.set_title('FruitNet - Training Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    # Right: Accuracy with target line
    ax2.plot(epochs_range, train_accs, 'b-', linewidth=2, label='Train')
    ax2.plot(epochs_range, val_accs, 'r-', linewidth=2, label='Validation')
    ax2.axhline(y=92, color='g', linestyle='--', linewidth=2, label='Target')
    ax2.set_title('FruitNet - Accuracy', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])

    plt.tight_layout()
    filename = 'FruitNet_training.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n[SAVED] {filename}")
    plt.close()


# ==================== Main ====================

def main():
    print("\n" + "="*80)
    print("Fruit Classification Pipeline - All-in-One")
    print("FruitNet + TensorFlow/Keras (PyTorch-style data processing)")
    print("="*80)

    train_gen, val_gen, test_gen = load_datasets()

    model = build_model(num_classes=NUM_CLASSES)
    compile_model(model, learning_rate=LEARNING_RATE)

    history, timestamp = train_model(model, train_gen, val_gen, epochs=NUM_EPOCHS)

    test_acc = evaluate_model(model, test_gen)

    plot_training_curves(history)

    results = {
        'model_name': 'FruitNet',
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'test_accuracy': float(test_acc),
        'timestamp': timestamp
    }

    results_path = f'FruitNet_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("TRAINING COMPLETED - FRUITNET")
    print("="*80)
    print(f"\nFruitNet:")
    print(f"  Best Validation Accuracy: {results['best_val_accuracy']*100:.2f}%")
    print(f"  Test Accuracy: {results['test_accuracy']*100:.2f}%")

    print(f"\n[SAVED] {results_path}")

    print("\n" + "="*80)
    print("DONE! Check FruitNet_training.png for training curves.")
    print("="*80)


if __name__ == '__main__':
    main()
