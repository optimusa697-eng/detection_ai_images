\# Detection of AI-Generated Images Using CNN



A Deep Learning framework implemented in Python using TensorFlow/Keras to distinguish genuine real-world images from AI-generated images produced by modern Generative Adversarial Networks (GANs) and generative AI tools (such as Leonardo.AI).



\---



\## 📌 Project Overview



With the rapid emergence of advanced GANs and AI tools, synthetic content poses significant challenges to digital trust and security. This project implements a dedicated \*\*Convolutional Neural Network (CNN)\*\* model designed specifically to detect synthetic artifacts and prevent image forgery, impersonation, and misinformation.



\### Key Highlights

\* \*\*Target Classes\*\*: `Real` vs. `AI-Generated`

\* \*\*Dataset Distribution\*\*:

&#x20; \* \*\*Training Set\*\*: 1,278 images

&#x20; \* \*\*Validation Set\*\*: 345 images

\* \*\*Performance\*\*: Achieves high sensitivity and target validation accuracy (up to \*\*96%\*\*).



\---



\## 🏗️ Model Architecture



The custom CNN architecture consists of 4 sequential convolutional blocks, flattening, dense layers, and dropout regularization:



1\. \*\*Input Layer\*\*: `128 x 128 x 3` RGB image inputs.

2\. \*\*Conv Block 1\*\*: 32 filters (`3x3`, ReLU activation) + Max Pooling (`2x2`).

3\. \*\*Conv Block 2\*\*: 64 filters (`3x3`, ReLU activation) + Max Pooling (`2x2`).

4\. \*\*Conv Block 3\*\*: 128 filters (`3x3`, ReLU activation) + Max Pooling (`2x2`).

5\. \*\*Conv Block 4\*\*: 128 filters (`3x3`, ReLU activation) + Max Pooling (`2x2`).

6\. \*\*Dense Classifier\*\*: Flatten -> Dense Layer (`128` units, ReLU) -> Dropout (`0.5`).

7\. \*\*Output Layer\*\*: Dense Layer (`1` unit, Sigmoid activation).



\---



\## 🚀 Getting Started



\### Prerequisites



Ensure you have Python 3.8+ installed along with the required libraries:



```bash

pip install tensorflow numpy matplotlib seaborn scikit-learn

