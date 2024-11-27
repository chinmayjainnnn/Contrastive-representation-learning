# **Contrastive Representation Learning and For image classification**

This repository contains implementations of Logistic Regression, Softmax Regression, and Contrastive Representation Learning using Python and PyTorch. The project focuses on feature extraction, classification, and embedding generation using supervised and contrastive learning techniques. 

---

## **Contents**
1. [Features](#features)
2. [Setup and Dependencies](#setup-and-dependencies)
3. [How to Run](#how-to-run)
4. [Implementation Details](#implementation-details)
5. [Files and Directory Structure](#files-and-directory-structure)
6. [Results](#results)
7. [Plots](#plots)

---

## **Features**

### **Logistic and Softmax Regression**
- Binary classification using logistic regression.
- Multi-class classification using softmax regression.
- Gradient-based optimization with L2 regularization and gradient clipping.

### **Contrastive Representation Learning**
- AlexNet-based encoder to extract feature embeddings.
- Triplet loss to learn meaningful embeddings.
- Fine-tuning using both linear classifiers and neural networks.
- Visualization of embeddings with t-SNE plots.

---

## **Setup and Dependencies**

### **1. Clone the Repository**
```bash
git clone https://github.com/chinmayjainnnn/Contrastive-representation-learning
```



#### **Key Dependencies:**
- Python 3.8+
- Numpy
- PyTorch (1.10+)
- Matplotlib
- Scikit-learn

---

## **How to Run**

### **1. Data Setup**
- **CIFAR-10 Dataset**:
  - Download the dataset [here](https://www.cs.toronto.edu/~kriz/cifar.html) and place it in the `data/` directory.
  - Ensure the directory structure is as follows:
    ```
    data/
    ├── cifar-10-batches-py/
    ├── train/
    └── test/
    ```

### **2. Logistic and Softmax Regression**
1. **Logistic Regression**:
    - Run the following command for binary classification:
      ```bash
      python LogisticRegression/main.py --mode logistic --num_iters 1000 --lr 0.01 --batch_size 256
      ```
2. **Softmax Regression**:
    - Run for multi-class classification:
      ```bash
      python LogisticRegression/main.py --mode softmax --num_iters 1000 --lr 0.01 --batch_size 256
      ```

### **3. Contrastive Representation Learning**
1. **Train Encoder with Contrastive Loss**:
    ```bash
    python ContrastiveRepresentation/main.py --mode cont_rep --num_iters 5000 --batch_size 1024 --lr 0.001
    ```
2. **Fine-Tune Classifier**:
   - **Linear Classifier**:
     ```bash
     python ContrastiveRepresentation/main.py --mode fine_tune_linear --num_iters 2000
     ```
   - **Neural Network Classifier**:
     ```bash
     python ContrastiveRepresentation/main.py --mode fine_tune_nn --num_iters 2000
     ```

---

## **Implementation Details**

### **Logistic and Softmax Regression**
- Implements gradient descent with:
  - L2 regularization.
  - Gradient clipping for stability.
- Custom loss functions:
  - Binary Cross-Entropy for logistic regression.
  - Categorical Cross-Entropy for softmax regression.

### **Contrastive Representation Learning**
1. **Encoder**:
   - AlexNet-style architecture.
   - Batch normalization, ReLU activation, and adaptive pooling.
2. **Triplet Loss**:
   - Encourages embeddings of similar data points to be close and dissimilar ones to be far apart.
3. **Visualization**:
   - t-SNE plots to visualize clusters in the embedding space.

---

## **Files and Directory Structure**
```
├── LogisticRegression/
│   ├── model.py         # Implements logistic and softmax regression models.
│   ├── train_utils.py   # Training utilities (loss, accuracy, training loop).
│   ├── main.py          # Main script for training and testing.
├── ContrastiveRepresentation/
│   ├── model.py         # Encoder and classifier architecture.
│   ├── train_utils.py   # Utilities for contrastive learning.
│   ├── pytorch_utils.py # PyTorch device management.
│   ├── main.py          # Main script for training and fine-tuning.
├── utils.py             # Data loading and preprocessing.
├── requirements.txt     # Python dependencies.
└── README.md            # Project documentation.
```

---

## **Results**

### **Logistic and Softmax Regression**
- **Binary Classification**:
  - Achieved >80% accuracy on the CIFAR-10 dataset.
- **Multi-class Classification**:
  - Validation accuracy: ~36.89% (softmax regression).

### **Contrastive Learning**
- **t-SNE Visualization**:
  - Clear cluster formations for different classes.
- **Fine-Tuning Accuracy**:
  - Neural network classifier: **82.38%** validation accuracy.

---


## **Plots**



### **t-SNE After Training**
![t-SNE After Training](plots/tsne.png)

---
