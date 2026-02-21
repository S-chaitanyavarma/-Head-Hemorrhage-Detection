Intracranial Hemorrhage Detection using Deep Learning

 📌 Overview

This repository contains a deep learning pipeline designed to detect and classify **Intracranial Hemorrhage (ICH)** from head CT scans. The model performs multi-label classification to identify five specific subtypes: **Intraparenchymal, Intraventricular, Subarachnoid, Subdural, and Epidural.**

 🚀 Features

* **DICOM Processing:** Automated windowing (Brain, Subdural, and Bone windows) to enhance bleed visibility.
* **Deep Learning Architecture:** Utilizes [Insert Model, e.g., ResNet50/EfficientNet] for high-accuracy feature extraction.
* **Explainable AI:** Integrated **Grad-CAM** visualizations to highlight medical areas of concern for radiologists.
* **High Performance:** Optimized to handle class imbalance common in medical datasets.

 🛠️ Installation

```bash
git clone https://github.com/yourusername/head-hemorrhage-detection.git
cd head-hemorrhage-detection
pip install -r requirements.txt

```

 📊 Dataset

This project primarily utilizes the **RSNA Intracranial Hemorrhage Detection** dataset, consisting of over 800,000 DICOM images.

> **Note:** Data must be downloaded separately from Kaggle due to licensing.

 📈 Results

The model achieves an **AUC-ROC of 0.XX** and an **F1-Score of 0.XX**. Detailed confusion matrices and loss curves can be found in the `/results` folder.

 📂 Project Structure

* `/data`: Scripts for data augmentation and preprocessing.
* `/models`: Saved model weights and architecture definitions.
* `/notebooks`: Jupyter notebooks for EDA and initial training.
* `train.py`: Main script to start model training.
* `inference.py`: Script to run predictions on new CT slices.
