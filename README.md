# 🫁 Multi-Class Lung Disease Classification using CNNs + Vision Transformer

This deep learning project classifies chest X-ray images into five categories: COVID-19, Bacterial Pneumonia, Viral Pneumonia, Tuberculosis, and Normal. We compare multiple CNN models (ResNet50, DenseNet121, EfficientNetB3) and a Vision Transformer (ViT), and combine them using a soft-voting ensemble.

---
## 1. Dataset Used
Dataset Name: Lungs Disease Dataset (4 Types)
Source: Kaggle
Link:  https://www.kaggle.com/datasets/omkarmanohardalvi/lungs-disease-dataset-4-types

## 📊 Summary

- Models used: ResNet50, DenseNet121, EfficientNetB3, ViT
- Ensemble method: Soft voting
- Evaluation: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Explainability: Grad-CAM visualizations
- Final Ensemble Accuracy: **92.49%**

---

## 📂 Files

| File | Description |
|------|-------------|
| `EDA.ipynb` | Exploratory data analysis |
| `Resnet50_Model.ipynb` | Training ResNet50 |
| `DensNet121_Model.ipynb` | Training DenseNet121 |
| `EfficientNetB3_Model.ipynb` | Training EfficientNetB3 |
| `ViT_Model.ipynb` | Vision Transformer model |
| `Ensemble_Model.ipynb` | Model ensembling and evaluation |

---

## 📈 Key Results

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| ResNet50 | 91.16% | 0.91 |
| ViT | 86.12% | 0.86 |
| **Ensemble** | **92.49%** | **0.92** |

---
 ## 📊 ROC Curve Analysis

To further evaluate the classifier's performance across all five lung disease classes, ROC (Receiver Operating Characteristic) curves were generated for each label.

Each curve plots the **True Positive Rate (Sensitivity)** against the **False Positive Rate**, and the **AUC (Area Under Curve)** provides a summary score of how well each class is separated.
ROC Curve screenshots added to the images folder*


## 🧠 Explainability with Grad-CAM

*Grad-CAM heatmaps were used to highlight lung regions influencing model decisions are added in images section*


---

## 📦 Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras torch torchvision timm opencv-python
```            

---

## 👨‍💻 Author

**Vamshi Krishna Kondra**  
📍 MSc in Data Analytics – Dublin Business School (2025)  
🔗 [GitHub](https://github.com/VamshiKondra77)  
🔗 [LinkedIn](https://www.linkedin.com/in/vamshi-krishna-kondra-5a124a216/)


