
# 🌿 Plant Disease Detection using ResNet18 (PyTorch)

## 📌 Project Overview
This project is an end-to-end deep learning pipeline for detecting plant diseases using **ResNet18**, transfer learning, fine-tuning, Grad‑CAM explainability, and full evaluation. It demonstrates a production‑quality structure suitable for real-world ML applications and GitHub portfolios.

## 📂 Dataset

### Dataset Source
Kaggle Dataset: **New Plant Diseases Dataset**  
Contains 38 classes of healthy and diseased plant leaves.

🔗 **Dataset Link:**  
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset


### Dataset Structure
```
data/
 ├── train/
 ├── valid/
 └── test/
```

## 🧠 Model Architecture
### Why ResNet18?
- Lightweight & efficient  
- Performs well on medium-sized datasets  
- Easy to fine‑tune  
- Strong baseline for image classification tasks  

### Transfer Learning Strategy
- Load pretrained ImageNet weights  
- Freeze base layers  
- Replace the fully connected layer  

### Fine-Tuning Strategy
- Unfreeze **layer4** for deeper learning  
- Train using **OneCycleLR** for improved convergence  

## 🛠️ Setup & Installation
```
pip install torch torchvision matplotlib scikit-learn
```
## 📁 Project Structure (Collapsible Table)

<details>
<summary><b>models/</b></summary>

| File Name |
|----------|
| plant_disease_checkpoint.pth |
| plant_disease_model_full_cpu.pt |
| plant_disease_model_onecycle.pth |
| plant_disease_model_torchscript.pt |
| plant_disease_resnet18.pth |
| plant_disease_resnet18_full.pt |

</details>

<details>
<summary><b>results/</b></summary>

| File Name |
|----------|
| classification_report.json |
| final_test_accuracy.txt |
| gradcam_5.png |
| gradcam_sample.png |
| model_predictions.csv |
| normalized_confusion_matrix.png |
| per_class_accuracies.png |
| per_class_accuracy.csv |
| training_curve_clean.png |
| training_validation_curve_fixed.png |

</details>

<details>
<summary><b>metadata/</b></summary>

| File Name |
|----------|
| augmentation_config.txt |
| class_labels.json |
| class_names.json |
| confusion_matrix.npy |
| gradcam_info.txt |
| onecycle_log.txt |
| training_history.csv |

</details>

<details>
<summary><b>inference_samples/</b></summary>

| File Name |
|----------|
| random_sample_prediction.txt |
| top3_prediction.txt |

</details>

<details>
<summary><b>src/</b></summary>

| File Name |
|----------|
| config.py |
| dataset.py |
| eval.py |
| gradcam.py |
| inference.py |
| model.py |
| train.py |
| transforms.py |

</details>

<details>
<summary><b>notebook/</b></summary>

| File Name |
|----------|
| plant-diseases-detection.ipynb |

</details>


---

## 🔧 Training Pipeline
### Base Training (Frozen Layers)
- Train classifier head for 5 epochs  
- Save base model + training logs  

### Fine-Tuning (Unfreezing Layer4)
- Allow deeper layers to update  
- Higher accuracy and robustness  

### OneCycleLR Training
- Fast, stable convergence  
- Produces the best performing model  

## 📊 Training Results
### Training & Validation Curves
Located in:  
`results/training_curve_clean.png`  
`results/training_validation_curve_fixed.png`

### Accuracy Scores
- Final Validation Accuracy: Logged in `training_history.csv`  

## 🧪 Evaluation
### Classification Report
Saved at:  
`results/classification_report.json`

### Confusion Matrix
`results/confusion_matrix.png`

### Normalized Confusion Matrix
`results/normalized_confusion_matrix.png`

### Per-Class Accuracy
`results/per_class_accuracies.png`  
`results/per_class_accuracy.csv`

## 🔍 Explainability (Grad-CAM)
### Single Image Grad-CAM
`results/gradcam_sample.png`

### Batch Grad-CAM
`results/gradcam_5.png`

### Interpretation
Grad‑CAM highlights important leaf regions contributing to predictions, improving transparency and trust.

## 🧾 Inference
### Single Image Prediction
```
python inference.py --image sample.jpg
```

### Top-3 Predictions
Stored in:  
`inference_samples/top3_prediction.txt`

### Batch Predictions
`inference_samples/batch_predictions.json`

### Random Sample Prediction
`inference_samples/random_sample_prediction.txt`

## 💾 Saved Models & Artifacts
### Model Formats
- `.pth` → PyTorch model weights  
- `.pt` → Full model (CPU)  
- TorchScript model → deployable version  

### Metadata Files
- `class_labels.json`  
- `class_names.json`  
- `augmentation_config.txt`  

### Logs & Reports
- `training_history.csv`  
- `onecycle_log.txt`  

## 🚀 How to Run the Project
### Run on Kaggle Notebook
1. Upload repository  
2. Enable GPU (T4)  
3. Run notebook:  
   `notebook/plant-diseases-detection.ipynb`

### Run Locally (Optional)
```
python src/train.py
python src/eval.py
python src/inference.py
```

## 🔮 Future Improvements
- Add EfficientNet / ConvNeXt models  
- Add FastAPI deployment  
- Add ONNX export  
- Add mobile app inference  

## 📜 License
MIT License  

## 👤 Author
**Ankush Patil**  
Machine Learning & Deep Learning Engineer
