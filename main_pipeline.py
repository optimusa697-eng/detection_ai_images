import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------------------------------------------------
# 1. CONFIGURATION & HYPERPARAMETERS
# (Based on paper specifications: 1278 train images, 345 validation)
# -------------------------------------------------------------------
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 1  # Binary classification: Real (0) vs AI-Generated (1)

# Set vibrant color palette for seaborn/matplotlib
sns.set_theme(style="whitegrid")
COLOR_TRAIN = "#1f77b4"  # Deep Blue
COLOR_VAL = "#ff7f0e"    # Vibrant Orange
ACCENT_COLOR = "#2ca02c" # Emerald Green


# -------------------------------------------------------------------
# 2. SYNTHETIC DATA GENERATOR (For Standalone Execution)
# Creates dummy folder structure & images if actual dataset isn't loaded
# -------------------------------------------------------------------
def create_dummy_dataset(base_dir="dataset"):
    """Generates synthetic dataset structure matching paper split counts."""
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "validation")
    
    classes = ["real", "ai_generated"]
    
    if not os.path.exists(base_dir):
        print("[INFO] Creating dummy dataset structure for execution...")
        for cls in classes:
            os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
            os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
            
            # Generate dummy image files
            for i in range(639): # ~1278 total training images
                img = np.random.randint(0, 256, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
                plt.imsave(os.path.join(train_dir, cls, f"img_{i}.png"), img)
            
            for i in range(172): # ~345 total validation images
                img = np.random.randint(0, 256, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
                plt.imsave(os.path.join(val_dir, cls, f"img_{i}.png"), img)
        print("[INFO] Dummy dataset populated successfully.")


# -------------------------------------------------------------------
# 3. BUILD CNN ARCHITECTURE
# (Exact model from Paper Fig 2: 4 Conv+MaxPool, Flatten, Dense, Dropout, Sigmoid)
# -------------------------------------------------------------------
def build_cnn_model():
    """Builds customized Convolutional Neural Network architecture."""
    model = models.Sequential([
        # Convolution Layer 1 + Max Pooling
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), name="Conv1_3x3x32"),
        layers.MaxPooling2D((2, 2), name="MaxPool1_2x2"),
        
        # Convolution Layer 2 + Max Pooling
        layers.Conv2D(64, (3, 3), activation='relu', name="Conv2_3x3x64"),
        layers.MaxPooling2D((2, 2), name="MaxPool2_2x2"),
        
        # Convolution Layer 3 + Max Pooling
        layers.Conv2D(128, (3, 3), activation='relu', name="Conv3_3x3x128"),
        layers.MaxPooling2D((2, 2), name="MaxPool3_2x2"),
        
        # Convolution Layer 4 + Max Pooling
        layers.Conv2D(128, (3, 3), activation='relu', name="Conv4_3x3x128"),
        layers.MaxPooling2D((2, 2), name="MaxPool4_2x2"),
        
        # Flatten & Dense Classification Layers
        layers.Flatten(name="Flatten"),
        layers.Dense(128, activation='relu', name="Dense_128"),
        layers.Dropout(0.5, name="Dropout_0.5"),
        layers.Dense(1, activation='sigmoid', name="Output_Sigmoid")
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# -------------------------------------------------------------------
# 4. COLORFUL VISUALIZATIONS & ANALYSIS OUTPUT
# -------------------------------------------------------------------
def plot_training_metrics(history):
    """Plots colorful training vs validation accuracy & loss graphs."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy Plot
    axes[0].plot(epochs_range, acc, label='Training Accuracy', color=COLOR_TRAIN, linewidth=2.5, marker='o')
    axes[0].plot(epochs_range, val_acc, label='Validation Accuracy', color=COLOR_VAL, linewidth=2.5, marker='s')
    axes[0].set_title('Model Training & Validation Accuracy', fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xlabel('Epochs', fontsize=11)
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Loss Plot
    axes[1].plot(epochs_range, loss, label='Training Loss', color=COLOR_TRAIN, linewidth=2.5, marker='o')
    axes[1].plot(epochs_range, val_loss, label='Validation Loss', color=COLOR_VAL, linewidth=2.5, marker='s')
    axes[1].set_title('Model Training & Validation Loss', fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel('Epochs', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('accuracy_loss_curves.png', dpi=300)
    plt.show()

def plot_confusion_matrix_heatmap(y_true, y_pred):
    """Generates a vibrant heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=True,
                xticklabels=['Real', 'AI-Generated'],
                yticklabels=['Real', 'AI-Generated'],
                annot_kws={"size": 14, "weight": "bold"})
    
    plt.title('Confusion Matrix Heatmap Analysis', fontsize=14, fontweight='bold', pad=12)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix_heatmap.png', dpi=300)
    plt.show()

def generate_classification_analysis(y_true, y_pred):
    """Calculates and outputs accuracy, precision, recall, and F1 score."""
    report = classification_report(y_true, y_pred, target_names=['Real', 'AI-Generated'], output_dict=True)
    
    print("\n" + "="*50)
    print("         MODEL PERFORMANCE & METRICS ANALYSIS        ")
    print("="*50)
    print(f" overall Accuracy : {report['accuracy'] * 100:.2f}%")
    print(f" Precision        : {report['weighted avg']['precision']:.4f}")
    print(f" Recall           : {report['weighted avg']['recall']:.4f}")
    print(f" F1-Score         : {report['weighted avg']['f1-score']:.4f}")
    print("="*50 + "\n")


# -------------------------------------------------------------------
# 5. MAIN EXECUTION PIPELINE
# -------------------------------------------------------------------
def main():
    base_dir = "dataset"
    create_dummy_dataset(base_dir)
    
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "validation")

    # Data Augmentation & Normalization (1/255.0)
    train_datagen = ImageDataGenerator(rescale=1.0/255.0, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    # Initialize and display architecture
    model = build_cnn_model()
    model.summary()

    # Train Model
    print("\n[INFO] Starting CNN Training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator
    )

    # Visualizations & Performance Assessment
    plot_training_metrics(history)
    
    val_generator.reset()
    predictions = model.predict(val_generator)
    y_pred = (predictions > 0.5).astype(int).reshape(-1)
    y_true = val_generator.classes

    plot_confusion_matrix_heatmap(y_true, y_pred)
    generate_classification_analysis(y_true, y_pred)

if __name__ == "__main__":
    main()