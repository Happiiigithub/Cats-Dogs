# Assignment: Cats vs Dogs Classifier  
**Name:** Asutosha  
**Roll No.:** 2205281 

---

## Problem Statement
Build a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images as **cats** or **dogs**. The goal is to learn the basics of deep learning for image classification, including data preprocessing, CNN model building, training, and evaluation.

---

## Dataset
- Source: [Kaggle – Cats vs Dogs (small version)](https://www.kaggle.com/datasets/tongpython/cat-and-dog)  
- Structure: Two folders – `cats/` and `dogs/`.  
- Preprocessing:
  - Images resized to **128×128 pixels**.
  - Pixel values normalized to **0–1** (scratch CNN) or **–1 to 1** (transfer learning).
  - Dataset split into **80% training** and **20% validation** using a fixed random seed.

---

## Approach
1. Imported required libraries: **TensorFlow/Keras, NumPy, Matplotlib**.  
2. Preprocessed dataset: resize, normalize, augment (flip, rotation, zoom).  
3. Built two models:  
   - **Scratch CNN**:  
     - 3× Conv2D + ReLU + MaxPooling  
     - Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)  
   - **Transfer Learning (MobileNetV2)**:  
     - Pretrained ImageNet backbone (frozen initially)  
     - GlobalAveragePooling → Dropout → Dense(1, Sigmoid)  
     - Fine-tuned top layers for better accuracy.  
4. Compiled with:  
   - **Loss:** Binary Cross-Entropy  
   - **Optimizer:** Adam  
   - **Metric:** Accuracy  
5. Training: 10–12 epochs with **EarlyStopping**.  
6. Evaluation: Printed accuracy/loss and tested on 20 random validation images.  
   - Predictions shown with confidence scores; correct ones marked in green, wrong in red.

---

## Results
- **Scratch CNN:**  
  - Training Accuracy: ~98%  
  - Validation Accuracy: ~78%  
- **MobileNetV2 Transfer Learning (final model):**  
  - Training Accuracy: ~98.8%  
  - Validation Accuracy: ~97.8%  
- Best Validation Accuracy: ~97.8% achieved at epoch (from EarlyStopping).  

Plots of training/validation accuracy and loss show that MobileNetV2 generalized far better than the scratch CNN (less overfitting).

---

## Challenges
- Without normalization and correct class mapping, early models failed (producing very low accuracy).  
- Scratch CNN showed overfitting after ~6 epochs.  
- Transfer learning required correct preprocessing (rescaling to –1 to 1).  

---

## Learnings
- Gained hands-on practice with TensorFlow/Keras pipelines.  
- Understood the importance of proper preprocessing and class mapping.  
- Learned how **transfer learning** significantly improves results on small datasets.  
- Learned to use **EarlyStopping** and augmentation to fight overfitting.  

---

## Files Submitted
1. **Notebook:** `cats_vs_dogs_classifier.ipynb`  
2. **README.md / README.pdf**  
3. (Optional) Saved model file: `cats_dogs_model.h5`

---

## How to Run
1. Open the notebook in **Google Colab**.  
2. Mount Google Drive and make sure `archive.zip` path is correct.  
3. Run all cells sequentially.  
4. To use transfer learning (recommended), run the MobileNetV2 cell.  
5. Check evaluation results and prediction visualizations at the end.
6. Github Link : https://github.com/Happiiigithub/Cats-Dogs
