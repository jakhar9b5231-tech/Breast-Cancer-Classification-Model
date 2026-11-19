# Breast Cancer Classification Using Machine Learning

A college-level machine learning project designed to classify breast tumors as either **Benign (non-cancerous)** or **Malignant (cancerous)** using clinical cell nucleus measurements.

---

## 1. Project Introduction
Breast cancer is one of the most common cancers worldwide. Early and accurate detection plays a crucial role in patient prognosis and treatment planning. 

In this project, we apply fundamental machine learning techniques in Python to analyze digitized fine needle aspirate (FNA) biopsy data and classify breast tumors.

> **Academic Disclaimer:** This project is developed strictly for educational and coursework demonstration purposes. It is not intended for real-world clinical diagnosis.

---

## 2. Dataset Description
The project uses the Kaggle Wisconsin Diagnostic Breast Cancer dataset located at `data/data.csv`.

- **Total Samples (Rows):** 569
- **Total Columns:** 33 original columns
- **Features Used:** 30 numerical measurements (10 Mean, 10 Standard Error, 10 Worst/Largest)
- **Target Variable:** `diagnosis`
  - `B` (Benign): 357 samples (62.74%)
  - `M` (Malignant): 212 samples (37.26%)
- **Data Cleaning Performed:**
  - Dropped `id`: Arbitrary patient identifier without predictive value.
  - Dropped `Unnamed: 32`: Completely empty column containing 100% NaN values due to trailing commas in the CSV.
  - Verified 0 duplicate rows and 0 remaining missing values.

---

## 3. Libraries Used
- **Python 3.11**
- **Pandas**: Data loading, inspection, and manipulation.
- **NumPy**: Numerical operations.
- **Matplotlib & Seaborn**: Exploratory data analysis and visualization.
- **Scikit-learn**: Data splitting, feature scaling (`StandardScaler`), model training, and performance evaluation.
- **Streamlit**: Web interface for interactive demonstration.
- **Joblib**: Saving and loading trained model artifacts.

---

## 4. Main Steps of the Project
1. **Data Loading & Inspection:** Loaded `data/data.csv`, checked shape, column names, data types, missing values, and duplicates.
2. **Data Cleaning:** Removed non-predictive `id` column and empty `Unnamed: 32` column. Separated features ($X$) and target ($y$).
3. **Exploratory Data Analysis (EDA):** Generated target class distribution plots, correlation heatmaps, feature histograms, box plots, and scatter plots.
4. **Data Preprocessing:** 
   - 80:20 train-test split (`test_size=0.20, random_state=42, stratify=y`).
   - Standardized features for distance-based algorithms using `StandardScaler` (fitted only on training data to prevent data leakage).
5. **Model Training:** Trained three standard Scikit-learn classifiers:
   - Gaussian Naive Bayes (`GaussianNB()`, unscaled)
   - Logistic Regression (`LogisticRegression()`, scaled)
   - Support Vector Machine (`SVC()`, scaled)
6. **Model Evaluation & Comparison:** Evaluated models using Accuracy, Precision, Recall, and F1-score on Malignant (`M`) cases, along with Confusion Matrices.
7. **Model Serialization:** Exported the best performing model and preprocessing scaler to `model.joblib`.
8. **Prediction Program:** Created `predict.py` for direct command-line prediction.
9. **Interactive Web App:** Created `app.py` using Streamlit for viva presentation.

---

## 5. Models Trained & Evaluation Results

Tested on 114 test samples (72 Benign, 42 Malignant):

| Model | Accuracy | Precision (M) | Recall (M) | F1-Score (M) | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gaussian Naive Bayes** | 93.86% | 1.0000 | 0.8333 | 0.9091 | 7 |
| **Logistic Regression** | 96.49% | 0.9750 | 0.9286 | 0.9512 | 3 |
| **Support Vector Machine (SVM)** | **97.37%** | **1.0000** | **0.9286** | **0.9630** | **3** |

### Why False Negatives Matter:
In cancer screening, a **False Negative** means a malignant tumor is wrongly classified as benign. The patient would receive false reassurance while cancer progresses untreated. Therefore, achieving high **Recall** for the malignant class is critical.

---

## 6. How to Install Requirements

1. Clone or open the project folder in your terminal:
   ```bash
   cd Breast-Cancer-Classification
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## 7. How to Run the Training Script

Run the following command to load the dataset, generate the plots, train all models, and save the best model:

```bash
python train_model.py
```
*(On Windows with Python launcher: `py train_model.py`)*

This script outputs the dataset summary, training logs, evaluation metrics, and saves visual plots in the `plots/` folder.

---

## 8. How to Run the Prediction Program

To test predictions on single samples via command line:

```bash
python predict.py
```
*(On Windows: `py predict.py`)*

---

## 9. How to Start the Streamlit Application

To launch the web interface for viva or project demonstration:

```bash
streamlit run app.py
```
*(On Windows: `py -m streamlit run app.py`)*

The browser will open automatically displaying:
- Project overview & objectives
- Dataset inspection & statistics
- Exploratory data analysis charts
- Model evaluation & comparison metrics
- Interactive prediction tool with sample loading
