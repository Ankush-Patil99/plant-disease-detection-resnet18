![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-red)
![HuggingFace](https://img.shields.io/badge/Model%20Hosted%20On-HuggingFace-yellow)
![Kaggle](https://img.shields.io/badge/Run%20On-Kaggle-blue)

# 🌿 Plant Disease Detection using ResNet18 (PyTorch)

## 📌 Project Overview
This project is an end-to-end deep learning pipeline for detecting plant diseases using **ResNet18**, transfer learning, fine-tuning, Grad‑CAM explainability, and full evaluation. It demonstrates a production‑quality structure suitable for real-world ML applications and GitHub portfolios.

## 📂 Dataset

### Dataset Source
Kaggle Dataset: **New Plant Diseases Dataset**  
Contains 38 classes of healthy and diseased plant leaves.

train/ → 54303 images  
valid/ → 6977 images  
test/ → 3498 images  


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
## 📁 Project Structure

Note: Each file includes a **Click Here** placeholder where users can manually insert Hugging Face links.

<details>
<summary><b>models/</b></summary>

| File Name | Open |
|----------|------|
| plant_disease_checkpoint.pth | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/models/plant_disease_checkpoint.pth) |
| plant_disease_model_full_cpu.pt | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/models/plant_disease_model_full_cpu.pt) |
| plant_disease_model_onecycle.pth | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/models/plant_disease_model_onecycle.pth) |
| plant_disease_model_torchscript.pt | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/models/plant_disease_model_torchscript.pt) |
| plant_disease_resnet18.pth | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/models/plant_disease_resnet18.pth) |
| plant_disease_resnet18_full.pt | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/models/plant_disease_resnet18_full.pt) |

</details>

<details>
<summary><b>results/</b></summary>

| File Name | Open |
|----------|------|
| classification_report.json | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/results/classification_report.json) |
| final_test_accuracy.txt | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/results/final_test_accuracy.txt) |
| gradcam_5.png | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/results/gradcam_5.png) |
| model_predictions.csv | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/results/model_predictions.csv) |
| normalized_confusion_matrix.png | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/results/normalized_confusion_matrix.png) |
| per_class_accuracies.png | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/results/per_class_accuracies.png) |
| per_class_accuracy.csv | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/results/per_class_accuracy.csv) |
| training_curve_clean.png | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/results/training_curve_clean.png) |
| training_validation_curve_fixed.png | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/results/training_validation_curve_fixed.png) |

</details>

<details>
<summary><b>metadata/</b></summary>

| File Name | Open |
|----------|------|
| augmentation_config.txt | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/metadata/augmentation_config.txt) |
| class_labels.json | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/metadata/class_labels.json) |
| class_names.json | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/metadata/class_names.json) |
| confusion_matrix.npy | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/metadata/confusion_matrix.npy) |
| gradcam_info.txt | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/metadata/gradcam_info.txt) |
| onecycle_log.txt | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/metadata/onecycle_log.txt) |
| training_history.csv | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/metadata/training_history.csv) |

</details>

<details>
<summary><b>inference_samples/</b></summary>

| File Name | Open |
|----------|------|
| random_sample_prediction.txt | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/inference_samples/random_sample_prediction.txt) |
| top3_prediction.txt | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/inference_samples/top3_prediction.txt) |

</details>

<details>
<summary><b>src/</b></summary>

| File Name | Open |
|----------|------|
| config.py | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/src/config.py) |
| dataset.py | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/src/dataset.py) |
| eval.py | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/src/eval.py) |
| gradcam.py | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/src/gradcam.py) |
| inference.py | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/src/inference.py) |
| model.py | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/src/model.py) |
| train.py | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/src/train.py) |
| transforms.py | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/src/transforms.py) |

</details>

<details>
<summary><b>notebook/</b></summary>

| File Name | Open |
|----------|------|
| plant-diseases-detection.ipynb | [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/notebook/plant-diseases-detection.ipynb) |

</details>


---

## 🔧 Training Pipeline

### **Base Training (Frozen Layers)**
- Trained only the classifier head for **5 epochs**
- Backbone layers kept frozen for stable initial learning
- Saved:
  - `plant_disease_resnet18.pth`
  - `training_history.csv`

### **Fine-Tuning (Unfreezing Layer4)**
- Unfroze **layer4** of ResNet18 for deeper feature extraction  
- Improved model generalization and robustness  
- Saved fine-tuned weights:
  - `plant_disease_model_finetuned.pth`

### **OneCycleLR Training**
- Applied **OneCycleLR** scheduling for faster, smoother convergence  
- Achieved the *best* validation accuracy among all training stages  
- Saved:
  - `plant_disease_model_onecycle.pth`
  - `onecycle_log.txt`

---

## 📊 Training Results

### **Training & Validation Curves**
All learning curves (training, validation, combined) are available in the **results folder**:

🔗 **Training Curves & Metrics:**  
https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/tree/main/plant-disease-detection/results

Files include:
- `training_curve_clean.png`
- `training_validation_curve_fixed.png`

### **Accuracy Scores**
- Final validation accuracy recorded in:  
  **`training_history.csv`**  
  (Located inside `metadata/` folder)

🔗 Metadata Folder:  
https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/tree/main/plant-disease-detection/metadata


## 🧪 Evaluation

### **Classification Report**
Stored at: **[Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/results/classification_report.json)**  
File: `results/classification_report.json`

### **Confusion Matrix**
Stored at: **[Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/metadata/confusion_matrix.npy)**  
File: `results/confusion_matrix.png`


### **Per-Class Accuracy**
Stored at: **[Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/tree/main/plant-disease-detection/results)**  
Files:
- `results/per_class_accuracies.png`
- `results/per_class_accuracy.csv`


---

## 🔍 Explainability (Grad-CAM)

### **Batch Grad-CAM (5 Samples)**
Stored at: **[Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/results/gradcam_5.png)**  
File: `results/gradcam_5.png`

### **Interpretation**
Grad-CAM highlights the critical image regions influencing the model’s decision, helping validate that the network focuses on the diseased portions of leaves and improving overall model transparency and trust.

## 🧾 Inference

### Single Image Prediction
```bash
python inference.py --image sample.jpg
```

### Top-3 Predictions  
<details>
<summary>Stored at (Click to expand)</summary>

- **top3_prediction.txt**  
  👉 [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/inference_samples/top3_prediction.txt)

</details>

### Batch Predictions  
<details>
<summary>Stored at (Click to expand)</summary>

- **batch_predictions.json**  
  👉 *(Not uploaded — add if needed)*

</details>

### Random Sample Prediction  
<details>
<summary>Stored at (Click to expand)</summary>

- **random_sample_prediction.txt**  
  👉 [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/inference_samples/random_sample_prediction.txt)

</details>


## 💾 Saved Models & Artifacts
### Model Formats  
<details>
<summary>Click to expand</summary>

- **plant_disease_checkpoint.pth**  
  👉 [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/models/plant_disease_checkpoint.pth)

- **plant_disease_model_full_cpu.pt**  
  👉 [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/models/plant_disease_model_full_cpu.pt)

- **plant_disease_model_onecycle.pth**  
  👉 [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/models/plant_disease_model_onecycle.pth)

- **plant_disease_model_torchscript.pt**  
  👉 [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/models/plant_disease_model_torchscript.pt)

- **plant_disease_resnet18.pth**  
  👉 [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/models/plant_disease_resnet18.pth)

- **plant_disease_resnet18_full.pt**  
  👉 [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/models/plant_disease_resnet18_full.pt)

</details>

### Metadata Files  
<details>
<summary>Click to expand</summary>

- **class_labels.json**  
  👉 [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/metadata/class_labels.json)

- **class_names.json**  
  👉 [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/metadata/class_names.json)

- **augmentation_config.txt**  
  👉 [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/metadata/augmentation_config.txt)

</details>

### Logs & Reports  
<details>
<summary>Click to expand</summary>

- **training_history.csv**  
  👉 [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/metadata/training_history.csv)

- **onecycle_log.txt**  
  👉 [Click Here](https://huggingface.co/ankpatil1203/plant-disease-detection-resnet18/blob/main/plant-disease-detection/metadata/onecycle_log.txt)

</details>

---
## 🚀 How to Run the Project (Kaggle Recommended)

### ✅ 1 — Open Kaggle Notebook
- Create a new Kaggle Notebook  
- Set Accelerator to **GPU (T4)**  

### ✅ 2 — Clone Your GitHub Repository
```
!git clone https://github.com/<your-username>/plant-disease-detection
```

### ✅ 3 — Install Dependencies
```
!pip install torch torchvision huggingface_hub pillow scikit-learn matplotlib
```

### ✅ 4 — Open and Run the Notebook
```
notebook/plant-diseases-detection.ipynb
```

➡ Running all cells will automatically:
- Fetch trained models from Hugging Face  
- Load dataset  
- Train / evaluate / run inference  
- Save outputs into results/ and inference_samples/  

---

## 🔮 Future Improvements
- Add EfficientNet / ConvNeXt models  
- Add FastAPI deployment  
- Add ONNX export  
- Add mobile app inference  

## 📜 License
MIT License  

## 👤 Author
**Ankush Patil**  
Machine Learning & NLP Engineer  
📧 **Email**: ankpatil1203@gmail.com  
💼 **LinkedIn**: www.linkedin.com/in/ankush-patil-48989739a  
🌐 **GitHub**: https://github.com/Ankush-Patil99  

