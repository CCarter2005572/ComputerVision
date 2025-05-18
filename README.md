# Pneumonia Detection from Chest X-ray Images

**Author:** Cameron Carter
**Coursework:** CM4126 – Computer Vision
**Dataset:** [Kaggle - Chest X-ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## 📘 Project Overview

This project explores the application of deep learning to classify **pneumonia** from chest X-ray images. Pneumonia is a serious lung infection that causes inflammation and fluid buildup, making early detection vital. By leveraging both **custom convolutional neural networks (CNNs)** and **transfer learning**, this notebook demonstrates a complete pipeline for building, training, and evaluating models for binary image classification (PNEUMONIA vs NORMAL).

---

## ✅ Key Features and Strengths

* **End-to-End Deep Learning Pipeline:**
  From data loading and preprocessing to model training and evaluation — all steps are clearly documented and executable in Google Colab.

* **Multiple Model Approaches:**

  * A **custom-built CNN** is developed as a baseline model.
  * Two **pre-trained models (VGG16 and InceptionV3)** are fine-tuned using transfer learning for performance improvement.

* **Performance Optimization:**

  * Use of **data augmentation** to reduce overfitting and increase generalizability.
  * **Early stopping** and **learning rate reduction** strategies are applied for training efficiency.

* **Model Explainability with Grad-CAM:**

  * Integrates **Grad-CAM** heatmaps to provide visual insights into what the models are learning — a valuable tool for medical imaging applications.

* **Strong Classification Performance:**

  * Achieves high accuracy and well-separated confusion matrices using pre-trained networks.
  * Includes evaluation metrics like **accuracy**, **precision**, **recall**, and **F1-score**.

---

## 🛠 Technologies Used

* **Python**
* **TensorFlow / Keras**
* **Google Colab**
* **Matplotlib & Seaborn (visualizations)**
* **Grad-CAM (explainability)**

---

## 📂 Folder Structure

```
.
├── Pneumonia_Classification_Notebook.ipynb
├── README.md
└── /chest_xray/
    ├── train/
    ├── val/
    └── test/
```

> The dataset folder must follow the structure as provided in the Kaggle link (separate folders for NORMAL and PNEUMONIA within each split).

---

## 📊 Results Summary

| Model       | Accuracy | F1-Score | Notes                            |
| ----------- | -------- | -------- | -------------------------------- |
| Custom CNN  | \~83%    | \~0.82   | Baseline model, minimal tuning   |
| VGG16       | \~92%    | \~0.91   | Fine-tuned with frozen base      |
| InceptionV3 | \~93%    | \~0.92   | Best performing, robust features |

Grad-CAM visualizations confirmed that the models focused on relevant lung regions in identifying pneumonia.

---

## 📌 Future Improvements

* Introduce cross-validation and hyperparameter tuning
* Experiment with more complex architectures like EfficientNet or ResNet
* Extend to multi-class classification (e.g., different pneumonia types)

---

## 📄 License & Usage

This project is for academic and educational purposes. The dataset is publicly available via Kaggle and is licensed accordingly.
