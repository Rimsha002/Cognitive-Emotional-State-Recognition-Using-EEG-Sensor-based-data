# Cognitive / Emotional State Recognition Using EEG Sensor Data

<img src="/results/DEAP-dataset.png" alt="neutral" width="800" height="800"/> 

##  Project Overview

This project presents a comparative study of multiple classification approaches for recognizing human cognitive and emotional states using EEG (Electroencephalogram) sensor data.

The goal is to analyze how different modeling paradigms perform when applied to the same dataset, including:

* Feature-engineering-based Classical machine learning
* Fully connected Deep Neural Networks trained on raw data
* Sequence-based Deep learning models (LSTM) trained on sequential data

The predicted emotional states are:

* Calm / Relaxed
* Neutral / Moderate
* Alert / Highly Aroused

---

## Objectives

* Perform preprocessing and feature engineering on EEG sensor data
* Generate emotional state labels using **unsupervised clustering (KMeans)**
* Train and evaluate **classical ML classifiers**
* Train and evaluate **deep learning models (DNN & LSTM)**
* Compare all approaches using Accuracy and Macro F1-Score
* Demonstrate user-level emotional state prediction

---

## Dataset Description

* **Data Source:** (https://www.kaggle.com/datasets/samnikolas/eeg-dataset)
* **Data Type:** EEG time-series / sequential data (tabular numerical features)
* **Sensors:** EEG electrodes (e.g., Fp1, AF3, F3, F7, …, O2)
* **Features:** 32 EGG Channel (Brain signal features extracted per channel)
* **Labels:**

  * Not originally provided
  * Generated using **KMeans clustering (3 clusters)**
* **Emotional State Mapping:**

  * Cluster 0 → Calm / Relaxed State
  * Cluster 1 → Neutral / Moderate State
  * Cluster 2 → Alert / Highly Aroused State
<img src="/results/Classes_by_KMeans.png" alt="neutral" width="600" height="600"/>
> ⚠️ Important Note
> These are pseudo-labels, used strictly for research comparison and benchmarking, not real clinical diagnosis.

---

## Kaggle Notebooks (Full Implementation)

The complete, runnable, and documented code for this project is available on Kaggle.

* Notebook Link:
(https://www.kaggle.com/code/rimmajeed/emotional-state-recognition-using-eeg-sensor-data)

The Kaggle notebooks include:
* Full preprocessing pipeline
* Classical ML baselines (Vervion 2)
* Fully connected DNN (Version 3)
* LSTM sequence model (Version 4)
* Visualizations, metrics, and analysis

> This GitHub repository serves as a project overview and research summary,  while Kaggle hosts the executable notebooks.


---

## Models Implemented

### 🔹 Classical Machine Learning

* Logistic Regression
* Support Vector Machine (RBF)
* Decision Tree
* Random Forest
* XGBoost

<img src="/results/confusion_matrix_Random_Forest.png" alt="neutral" width="400" height="400"/>

### 🔹 Deep Learning

* Fully Connected Neural Network (DNN)
  
<img src="/results/DNN__train-curve.png" alt="train curve" width="800" height="800"/>

<img src="/results/confusion_matrix_dnn.png" alt="Confusion matrix" width="400" height="400"/>

* Long Short-Term Memory Network (LSTM)
<img src="/results/confusion_matrix_lstm.png" alt="neutral" width="400" height="400"/>

---

## Methodology

### 1️⃣ Data Preprocessing

* Removal of duplicate rows
* Removal of non-numeric and empty columns
* Missing value handling using mean imputation
* Feature scaling using StandardScaler

### 2️⃣ Label Generation

* Since no ground-truth labels were provided:

  * **KMeans clustering (k=3)** was applied
  * Clusters were interpreted as emotional states

### 3️⃣ Model Training & Evaluation

* Train–test split (80/20)
* Evaluation metrics:

  * Accuracy
  * Precision (macro)
  * Recall (macro)
  * F1-score (macro)
<img src="/results/lstm_evalution.png" alt="neutral" width="600" height="600"/>

---

## Results Summary

| Approach                 | Model                   | Accuracy (%) | F1-Score (%)            |
| ------------------------ | ----------------------- | ------------ | ----------------------- |
| Feature Engineering + ML | Random Forest           | ~99.0        | ~98.6                   |
| Fully Connected DNN      | Raw Data                | ~99.26       | ~96.99                  |
| **Sequence-Based Model** | **LSTM**                | ~99.07       | ~95.87                  |

<img src="/results/model_confusion_m_comparison.png" alt="neutral" width="1000" height="1000"/>

---

## User-Level Prediction Example

The system supports **single-user EEG input** and outputs:

* Predicted emotional state
* Confidence scores

Example output:

```
Predicted Emotional State: Alert / Highly Aroused State
```

⚠️ Limitations
* Labels are machine-generated, not expert-annotated
* High accuracy does not imply real-world clinical reliability
* Dataset size is limited
* No cross-subject validation

 Future Work
* Class imbalance handling
* Real-time EEG emotion recognition interface
* Ground-truth labeled datasets
* Attention-based LSTM / Transformer models
* CNN-LSTM hybrid architectures

---

## Repository Structure

```
EEG-Emotional-State-Recognition/
│
├── notebooks
│   ( code available on the Kaggle notebook)
|
├── results/
│   ├── confusion_matrix.png
│
├── models/
│   └── best_lstm_model.h5
└── README.md
```

---


##  Technologies Used

* Python
* NumPy, Pandas
* Scikit-learn
* XGBoost
* TensorFlow / Keras
* Matplotlib

---

##  Author

**Rimsha Majeed**
Research-Oriented Machine Learning Project

---






