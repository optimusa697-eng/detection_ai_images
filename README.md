# AI Image Detection CNN — Simplified Google Colab Version

## Overview

This project is a simplified, user-input-friendly version of the uploaded `main_pipeline.py`.

It trains a Convolutional Neural Network (CNN) to perform binary image classification:

- `0` = Real
- `1` = AI-Generated

The notebook is designed to run in **Google Colab**.

## Files

- `AI_Image_Detection_CNN_Simplified_Colab.ipynb` — complete Colab notebook.
- `README.md` — project documentation.

## CNN architecture

The model follows the architecture of the uploaded program:

1. Conv2D: 32 filters, 3×3
2. MaxPooling: 2×2
3. Conv2D: 64 filters, 3×3
4. MaxPooling: 2×2
5. Conv2D: 128 filters, 3×3
6. MaxPooling: 2×2
7. Conv2D: 128 filters, 3×3
8. MaxPooling: 2×2
9. Flatten
10. Dense: 128 neurons
11. Dropout: 0.5
12. Dense: 1 neuron with sigmoid

The optimizer is Adam and the loss function is binary cross-entropy.

## Dataset format

For a real dataset, use:

```text
dataset/
├── train/
│   ├── real/
│   └── ai_generated/
└── validation/
    ├── real/
    └── ai_generated/
```

Put image files inside the four class folders.

## How to run in Google Colab

1. Open Google Colab.
2. Upload/open `AI_Image_Detection_CNN_Simplified_Colab.ipynb`.
3. Run the cells from top to bottom.
4. When asked for the number of epochs, enter a value such as `10`.
5. Choose:
   - `REAL` to upload a ZIP containing your dataset.
   - `DEMO` to create a small synthetic dataset for testing the pipeline.
6. After training, review:
   - training accuracy
   - validation accuracy
   - training loss
   - validation loss
   - confusion matrix
   - classification report
7. Upload an image in the final prediction cell to classify it.

## Dataset ZIP

The uploaded ZIP should contain either:

```text
dataset/
    train/
        real/
        ai_generated/
    validation/
        real/
        ai_generated/
```

or a top-level folder with the same four subfolders.

## Image preprocessing

Every image is resized to:

```text
128 × 128 × 3
```

Pixel values are normalized using:

```text
normalized_pixel = pixel / 255
```

Therefore the input range is approximately 0 to 1.

Training images also use horizontal flipping as augmentation.

## Prediction formula

The final CNN layer uses the sigmoid function:

```text
P(AI-Generated) = 1 / (1 + exp(-z))
```

where `z` is the value produced by the final neuron before sigmoid activation.

The class decision is:

```text
if P(AI-Generated) > 0.5:
    AI-Generated
else:
    Real
```

## Evaluation formulas

Let:

- TP = AI-generated images correctly predicted as AI-generated
- TN = Real images correctly predicted as Real
- FP = Real images incorrectly predicted as AI-generated
- FN = AI-generated images incorrectly predicted as Real

### Accuracy

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision

```text
Precision = TP / (TP + FP)
```

### Recall

```text
Recall = TP / (TP + FN)
```

### F1-score

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

## Understanding the confusion matrix

The confusion matrix has actual labels on one axis and predicted labels on the other.

Conceptually:

```text
                    Predicted
                 Real   AI-Generated
Actual Real       TN         FP
       AI         FN         TP
```

- `TN`: correct Real prediction.
- `TP`: correct AI-Generated prediction.
- `FP`: Real image incorrectly classified as AI-Generated.
- `FN`: AI-generated image incorrectly classified as Real.

## Understanding training graphs

### Accuracy graph

If training and validation accuracy increase and remain reasonably close, the model is learning patterns that also work on validation images.

If training accuracy becomes much higher than validation accuracy, this can indicate overfitting.

### Loss graph

Lower loss generally means the model's predictions are closer to the target labels.

A large gap between training and validation loss can also indicate overfitting.

## Single-image output

For an uploaded image, the notebook prints:

```text
Prediction: AI-Generated
P(AI-Generated) = 0.87
P(Real)         = 0.13
Reported confidence = 87.00%
```

The result is calculated directly from the sigmoid output and the 0.5 threshold.

The displayed confidence should not be interpreted as a guarantee. It is the model's probability-like output.

## Important limitation of DEMO mode

The original uploaded code creates random images when a dataset is not available.

This simplified notebook preserves that idea only as a pipeline-testing option. Random images do **not** contain meaningful visual differences between real and AI-generated images. Therefore, demo-mode accuracy is not a valid measure of a real AI-image detector.

For meaningful results, use a properly labeled real/AI-generated image dataset.

## Main output

After execution, the notebook provides:

1. CNN model summary.
2. Training/validation accuracy graph.
3. Training/validation loss graph.
4. Confusion matrix.
5. Accuracy.
6. Precision.
7. Recall.
8. F1-score.
9. Single-image prediction with probability.
10. Optional saved Keras model: `ai_image_detector.keras`.

## Source

The architecture and processing flow were simplified from the uploaded `main_pipeline.py`, including its 128×128 input size, four convolution/pooling blocks, Dense-128 layer, 0.5 dropout, sigmoid binary output, Adam optimizer, binary cross-entropy loss, image normalization, validation evaluation, confusion matrix, and classification metrics.
