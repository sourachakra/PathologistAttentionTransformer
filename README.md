# **Code for "Where and When Pathologists Focus Their Visual Attention While Grading Whole Slide Images of Cancer"**  

This repository contains code for predicting the **spatio-temporal dynamics** of pathologists' visual attention while analyzing whole slide images (WSIs) for tumor grading. Specifically, the project focuses on predicting:  
- **Attention heatmaps** (spatial distribution of attention).  
- **Attention scanpaths** (temporal sequences of gaze fixation).  

## **1⃣ Heatmap Prediction (PAT-H Model)**  
The **`heatmap_prediction`** directory contains the implementation for:  
1. Predicting **attention heatmaps** on WSIs.  
2. Extracting **feature encodings** of WSIs at multiple magnification levels using a model trained on the heatmap prediction task.  

### **🚀 Running Heatmap Prediction**  
Follow these steps to train and evaluate the **PAT-H model**:  

1. Navigate to the `heatmap_prediction` directory:  
   ```bash
   cd heatmap_prediction
   ```  
2. Install required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Download the **models and dataset** from [Google Drive](https://drive.google.com/file/d/15dHFFgLkwh_OmkdN3KdV8bbrLwDEAZ5B/view?usp=sharing).  
4. Extract the downloaded zip file and place the **`heatmap_drive`** directory into the corresponding paths within `heatmap_prediction`.  
5. Run the following commands:  
   - **Train the PAT-H model**:  
     ```bash
     python train.py
     ```  
   - **Evaluate a trained PAT-H model**:  
     ```bash
     python eval.py
     ```  

---

## **2⃣ Scanpath Prediction (PAT-S Model)**  
The **`scanpath_prediction`** directory contains the implementation for predicting **attention scanpaths** on WSIs. This model builds on the feature encodings extracted during the heatmap prediction stage.  

### **🚀 Running Scanpath Prediction**  
Follow these steps to train and evaluate the **PAT-S model**:  

1. Navigate to the `scanpath_prediction` directory:  
   ```bash
   cd scanpath_prediction
   ```  
2. Install required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Download the **models and dataset** from [Google Drive](https://drive.google.com/file/d/15dHFFgLkwh_OmkdN3KdV8bbrLwDEAZ5B/view?usp=sharing).  
4. Extract the downloaded zip file and place the **`scanpath_drive`** directory into the corresponding paths within `scanpath_prediction`.  
5. Run the following commands:  
   - **Train the PAT-S model**:  
     ```bash
     python train_sptransformer.py --hparams ./configs/coco_freeview_dense_SSL.json --dataset-root ./datasets/WSIs
     ```  
   - **Evaluate a trained PAT-S model**:  
     ```bash
     python train_sptransformer.py --hparams ./configs/coco_freeview_dense_SSL.json --dataset-root ./datasets/WSIs --eval-only
     ```  

---

### **📌 Notes**  
- Ensure all dependencies are installed as specified in `requirements.txt`.  
- The dataset and pre-trained model weights must be placed in the correct directories after extraction.  
- For further details on hyperparameter tuning, refer to the **configuration files** in the `configs/` directory.  

---
