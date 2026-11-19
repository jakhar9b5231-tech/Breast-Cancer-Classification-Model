# ==============================================================================
# Project: Breast Cancer Classification Using Machine Learning
# File: train_model.py
# Description: College-level machine learning script to load, explore, preprocess,
#              train, and evaluate Gaussian Naive Bayes, Logistic Regression,
#              and Support Vector Machine classifiers on the Wisconsin dataset.
# ==============================================================================

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Suppress minor library warnings for clean terminal presentation
warnings.filterwarnings("ignore")

# Set style for Matplotlib and Seaborn plots
sns.set_theme(style="whitegrid")

# Determine which step to run (default: run all steps)
selected_step = None
for arg in sys.argv[1:]:
    if arg.startswith("--step=") or arg.startswith("--part="):
        selected_step = int(arg.split("=")[1])
    elif arg in ["--step", "--part", "-s", "-p"] and len(sys.argv) > sys.argv.index(arg) + 1:
        selected_step = int(sys.argv[sys.argv.index(arg) + 1])


def should_print(step_num):
    """Returns True if the current step should print its output."""
    return selected_step is None or selected_step == step_num


# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)


# ------------------------------------------------------------------------------
# 1. DATA LOADING
# ------------------------------------------------------------------------------
if should_print(1):
    print("=" * 60)
    print("STEP 1: DATA LOADING")
    print("=" * 60)

# Locate dataset whether executed from root or subfolder
if os.path.exists("data/data.csv"):
    data_path = "data/data.csv"
elif os.path.exists("../data/data.csv"):
    data_path = "../data/data.csv"
else:
    data_path = "C:/breast cancer wisconsin/data/data.csv"

df = pd.read_csv(data_path)

if should_print(1):
    print(f"Dataset successfully loaded from: {data_path}")
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    print("--- First 5 Rows ---")
    print(df.head())

    print("\n--- Column Names ---")
    print(list(df.columns))

    print("\n--- Dataset Information ---")
    df.info()

    print("\n--- Statistical Summary ---")
    print(df.describe())

    print("\n--- Missing Values in Each Column ---")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    print(f"\nNumber of Duplicate Rows: {df.duplicated().sum()}")


# ------------------------------------------------------------------------------
# 2. DATA CLEANING
# ------------------------------------------------------------------------------
if should_print(2):
    print("\n" + "=" * 60)
    print("STEP 2: DATA CLEANING")
    print("=" * 60)

# Step 2a: Remove 'Unnamed: 32' because it contains only NaN values
if "Unnamed: 32" in df.columns:
    df = df.drop(columns=["Unnamed: 32"])
    if should_print(2):
        print("Dropped column 'Unnamed: 32' (all missing values).")

# Step 2b: Remove 'id' column because patient ID is an arbitrary identifier
if "id" in df.columns:
    df = df.drop(columns=["id"])
    if should_print(2):
        print("Dropped column 'id' (identifier column).")

# Check class distribution of target variable
if should_print(2):
    print("\nTarget Class Distribution ('diagnosis'):")
    print(df["diagnosis"].value_counts())
    print("\nClass Proportion (%):")
    print((df["diagnosis"].value_counts(normalize=True).round(4) * 100).to_string())

# Step 2c: Separate input features (X) and target variable (y)
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]
feature_names = list(X.columns)

if should_print(2):
    print(f"\nFeatures (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"Number of predictive features: {len(feature_names)}")


# ------------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------------------------
if should_print(3):
    print("\n" + "=" * 60)
    print("STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    # 3a. Count plot of the target classes (B vs M)
    plt.figure(figsize=(6, 4))
    palette = {"B": "#2ecc71", "M": "#e74c3c"}
    ax = sns.countplot(x=y, hue=y, palette=palette, legend=False)
    plt.title("Distribution of Diagnosis (Benign vs Malignant)", fontsize=13)
    plt.xlabel("Diagnosis (B = Benign, M = Malignant)", fontsize=11)
    plt.ylabel("Count", fontsize=11)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/target_distribution.png", dpi=150)
    plt.close()
    print("Saved: plots/target_distribution.png")

    # 3b. Correlation Heatmap of selected features (mean features)
    mean_features = [col for col in X.columns if "_mean" in col]
    plt.figure(figsize=(10, 8))
    corr_matrix = X[mean_features].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Heatmap (Mean Features)", fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/correlation_heatmap.png", dpi=150)
    plt.close()
    print("Saved: plots/correlation_heatmap.png")

    # 3c. Histograms of selected numerical features
    selected_features = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean"]
    plt.figure(figsize=(10, 8))
    for i, feature in enumerate(selected_features, 1):
        plt.subplot(2, 2, i)
        sns.histplot(df, x=feature, hue="diagnosis", kde=True, palette=palette, bins=25)
        plt.title(f"Histogram of {feature}", fontsize=11)
        plt.xlabel(feature)
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("plots/feature_histograms.png", dpi=150)
    plt.close()
    print("Saved: plots/feature_histograms.png")

    # 3d. Box plots for selected features
    plt.figure(figsize=(10, 8))
    for i, feature in enumerate(selected_features, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x="diagnosis", y=feature, hue="diagnosis", data=df, palette=palette, legend=False)
        plt.title(f"Box Plot: {feature} by Diagnosis", fontsize=11)
        plt.xlabel("Diagnosis")
        plt.ylabel(feature)
    plt.tight_layout()
    plt.savefig("plots/feature_boxplots.png", dpi=150)
    plt.close()
    print("Saved: plots/feature_boxplots.png")

    # 3e. Scatter plot of a key feature pair (radius_mean vs texture_mean)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x="radius_mean", y="texture_mean", hue="diagnosis", data=df, palette=palette, alpha=0.8)
    plt.title("Scatter Plot: Radius Mean vs Texture Mean", fontsize=13)
    plt.xlabel("Radius Mean")
    plt.ylabel("Texture Mean")
    plt.tight_layout()
    plt.savefig("plots/scatter_plot.png", dpi=150)
    plt.close()
    print("Saved: plots/scatter_plot.png")


# ------------------------------------------------------------------------------
# 4. DATA PREPROCESSING
# ------------------------------------------------------------------------------
if should_print(4):
    print("\n" + "=" * 60)
    print("STEP 4: DATA PREPROCESSING")
    print("=" * 60)

# Split dataset into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

if should_print(4):
    print(f"Training set: {X_train.shape[0]} samples (80%)")
    print(f"Testing set:  {X_test.shape[0]} samples (20%)")
    print("Train Class Distribution:")
    print(y_train.value_counts().to_dict())
    print("Test Class Distribution:")
    print(y_test.value_counts().to_dict())

# Feature Scaling using StandardScaler fitted ONLY on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

if should_print(4):
    print("\nStandardScaler fitted on training data and applied to train/test sets.")
    print("Gaussian Naive Bayes will use unscaled features.")
    print("Logistic Regression & Support Vector Machine will use scaled features.")


# ------------------------------------------------------------------------------
# 5. MACHINE LEARNING MODELS
# ------------------------------------------------------------------------------
if should_print(5):
    print("\n" + "=" * 60)
    print("STEP 5: TRAINING MACHINE LEARNING MODELS")
    print("=" * 60)

# Define the three classifiers
models = {
    "Gaussian Naive Bayes": {
        "model": GaussianNB(),
        "train_X": X_train,          # GaussianNB runs on unscaled data
        "test_X": X_test
    },
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "train_X": X_train_scaled,   # Scaled data
        "test_X": X_test_scaled
    },
    "Support Vector Machine": {
        "model": SVC(random_state=42),  # Scaled data
        "train_X": X_train_scaled,
        "test_X": X_test_scaled
    }
}

# Train each model
for name, config in models.items():
    if should_print(5):
        print(f"Training {name}...")
    config["model"].fit(config["train_X"], y_train)

if should_print(5):
    print("All models trained successfully.")


# ------------------------------------------------------------------------------
# 6. MODEL EVALUATION
# ------------------------------------------------------------------------------
if should_print(6):
    print("\n" + "=" * 60)
    print("STEP 6: MODEL EVALUATION")
    print("=" * 60)

results = []
confusion_matrices = {}
pos_class = "M"

for name, config in models.items():
    clf = config["model"]
    test_X = config["test_X"]
    y_pred = clf.predict(test_X)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=pos_class)
    rec = recall_score(y_test, y_pred, pos_label=pos_class)
    f1 = f1_score(y_test, y_pred, pos_label=pos_class)
    cm = confusion_matrix(y_test, y_pred, labels=["B", "M"])
    confusion_matrices[name] = cm

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })

    if should_print(6):
        print(f"\n--- {name} ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("\nConfusion Matrix (labels: [B, M]):")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Benign (B)", "Malignant (M)"]))

if should_print(6):
    # Plot Confusion Matrices side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Benign (B)", "Malignant (M)"],
                    yticklabels=["Benign (B)", "Malignant (M)"],
                    ax=axes[idx])
        axes[idx].set_title(f"CM: {name}", fontsize=11)
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrices.png", dpi=150)
    plt.close()
    print("Saved: plots/confusion_matrices.png")

    print("\n--- Clinical Relevance: Why False Negatives Matter ---")
    print("In breast cancer detection, a False Negative (FN) means a malignant tumor")
    print("was incorrectly predicted as benign. This leads to a missed cancer diagnosis")
    print("and delayed treatment, which can be fatal. Hence, high Recall for class 'M'")
    print("is critically important in this domain.")
    print("\n*DISCLAIMER: This project is for educational purposes and is not a")
    print("substitute for professional medical diagnosis or clinical healthcare.*")


# ------------------------------------------------------------------------------
# 7. MODEL COMPARISON & SELECTION
# ------------------------------------------------------------------------------
if should_print(7):
    print("\n" + "=" * 60)
    print("STEP 7: MODEL COMPARISON & SELECTION")
    print("=" * 60)

comparison_df = pd.DataFrame(results)

if should_print(7):
    print("\nModel Comparison Table:")
    print(comparison_df.to_string(index=False))

    # Plot accuracy comparison bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(comparison_df["Model"], comparison_df["Accuracy"], color=["#3498db", "#2ecc71", "#9b59b6"], width=0.5)
    plt.ylim(0.85, 1.0)
    plt.title("Model Accuracy Comparison", fontsize=13)
    plt.ylabel("Accuracy Score", fontsize=11)
    plt.xlabel("Model", fontsize=11)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/model_comparison.png", dpi=150)
    plt.close()
    print("Saved: plots/model_comparison.png")

    best_model_row = comparison_df.sort_values(by=["F1-Score", "Accuracy"], ascending=False).iloc[0]
    best_model_name = best_model_row["Model"]
    best_model_instance = models[best_model_name]["model"]

    print(f"\nSelected Best Model: {best_model_name}")
    print(f"Reason: Achieved highest F1-Score ({best_model_row['F1-Score']:.4f}) and Accuracy ({best_model_row['Accuracy']:.4f}) with minimal false negatives.")

    # Save trained model artifacts
    save_data = {
        "model": best_model_instance,
        "model_name": best_model_name,
        "scaler": scaler,
        "feature_names": feature_names,
        "comparison_table": comparison_df
    }
    joblib.dump(save_data, "model.joblib")
    print("\nModel and preprocessing artifacts saved to 'model.joblib'.")
    print("=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
