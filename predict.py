# ==============================================================================
# Project: Breast Cancer Classification Using Machine Learning
# File: predict.py
# Description: Simple prediction program that loads the saved model, validates
#              new input features against data.csv columns, and predicts the diagnosis.
# ==============================================================================

import os
import sys
import joblib
import pandas as pd

# Path to the trained model file
MODEL_FILE = "model.joblib"


def load_saved_model(filepath=MODEL_FILE):
    """Loads the trained model, scaler, and feature names saved by train_model.py."""
    if not os.path.exists(filepath):
        # Fallback check if located in parent or subfolder
        if os.path.exists(os.path.join("..", filepath)):
            filepath = os.path.join("..", filepath)
        elif os.path.exists(os.path.join("programs", filepath)):
            filepath = os.path.join("programs", filepath)
        else:
            raise FileNotFoundError(
                f"Model file '{filepath}' not found. Please run 'train_model.py' first."
            )
    return joblib.load(filepath)


def predict_tumor(input_dict):
    """
    Validates input features, scales them if necessary, and predicts the tumor class.
    
    Parameters:
        input_dict (dict): Dictionary with feature names as keys and measurements as values.
    
    Returns:
        prediction (str): 'B' for Benign or 'M' for Malignant.
        label (str): Full readable text label.
        model_name (str): Name of the model used for prediction.
    """
    artifacts = load_saved_model()
    model = artifacts["model"]
    model_name = artifacts["model_name"]
    scaler = artifacts["scaler"]
    expected_features = artifacts["feature_names"]

    # 1. Validate that all required features are present
    missing_features = [f for f in expected_features if f not in input_dict]
    if missing_features:
        raise ValueError(
            f"Input is missing {len(missing_features)} required feature(s): {missing_features[:3]}..."
        )

    # 2. Arrange features in the exact column order expected by the model
    input_df = pd.DataFrame([input_dict])[expected_features]

    # 3. Apply StandardScaler if model is Logistic Regression or Support Vector Machine
    if model_name in ["Logistic Regression", "Support Vector Machine"]:
        input_processed = scaler.transform(input_df)
    else:
        # Gaussian Naive Bayes was trained on unscaled features
        input_processed = input_df

    # 4. Predict diagnosis
    prediction = model.predict(input_processed)[0]
    
    label = "Malignant (Cancerous)" if prediction == "M" else "Benign (Non-Cancerous)"
    return prediction, label, model_name


# Preset test samples taken directly from Kaggle Wisconsin dataset (data/data.csv)
sample_malignant = {
    'radius_mean': 17.99, 'texture_mean': 10.38, 'perimeter_mean': 122.8,
    'area_mean': 1001.0, 'smoothness_mean': 0.1184, 'compactness_mean': 0.2776,
    'concavity_mean': 0.3001, 'concave points_mean': 0.1471, 'symmetry_mean': 0.2419,
    'fractal_dimension_mean': 0.07871, 'radius_se': 1.095, 'texture_se': 0.9053,
    'perimeter_se': 8.589, 'area_se': 153.4, 'smoothness_se': 0.006399,
    'compactness_se': 0.04904, 'concavity_se': 0.05373, 'concave points_se': 0.01587,
    'symmetry_se': 0.03003, 'fractal_dimension_se': 0.006193, 'radius_worst': 25.38,
    'texture_worst': 17.33, 'perimeter_worst': 184.6, 'area_worst': 2019.0,
    'smoothness_worst': 0.1622, 'compactness_worst': 0.6656, 'concavity_worst': 0.7119,
    'concave points_worst': 0.2654, 'symmetry_worst': 0.4601,
    'fractal_dimension_worst': 0.1189
}

sample_benign = {
    'radius_mean': 13.54, 'texture_mean': 14.36, 'perimeter_mean': 87.46,
    'area_mean': 566.3, 'smoothness_mean': 0.09779, 'compactness_mean': 0.08129,
    'concavity_mean': 0.06664, 'concave points_mean': 0.04781, 'symmetry_mean': 0.1885,
    'fractal_dimension_mean': 0.05766, 'radius_se': 0.2699, 'texture_se': 0.7886,
    'perimeter_se': 2.058, 'area_se': 23.56, 'smoothness_se': 0.008462,
    'compactness_se': 0.0146, 'concavity_se': 0.02387, 'concave points_se': 0.01315,
    'symmetry_se': 0.0198, 'fractal_dimension_se': 0.0023, 'radius_worst': 15.11,
    'texture_worst': 19.26, 'perimeter_worst': 99.7, 'area_worst': 711.2,
    'smoothness_worst': 0.144, 'compactness_worst': 0.1773, 'concavity_worst': 0.239,
    'concave points_worst': 0.1288, 'symmetry_worst': 0.2977,
    'fractal_dimension_worst': 0.07259
}


def run_malignant_test():
    print("\n--- Testing Sample 1 (Actual Dataset Row 0 | Expected: Malignant) ---")
    print(f"Key Features: radius_mean={sample_malignant['radius_mean']}, area_mean={sample_malignant['area_mean']}, concavity_mean={sample_malignant['concavity_mean']}")
    pred, label, model_used = predict_tumor(sample_malignant)
    print(f"Model Used:     {model_used}")
    print(f"Raw Prediction: {pred}")
    print(f"Result:         {label}")
    return pred == "M"


def run_benign_test():
    print("\n--- Testing Sample 2 (Actual Dataset Row 19 | Expected: Benign) ---")
    print(f"Key Features: radius_mean={sample_benign['radius_mean']}, area_mean={sample_benign['area_mean']}, concavity_mean={sample_benign['concavity_mean']}")
    pred, label, model_used = predict_tumor(sample_benign)
    print(f"Model Used:     {model_used}")
    print(f"Raw Prediction: {pred}")
    print(f"Result:         {label}")
    return pred == "B"


def run_validation_test():
    print("\n--- Testing Feature Validation & Error Handling ---")
    incomplete_sample = {'radius_mean': 14.5, 'texture_mean': 18.2}
    print(f"Passing incomplete input with only 2 features instead of required 30 features...")
    try:
        predict_tumor(incomplete_sample)
        print("FAILED: Did not catch missing features.")
    except ValueError as e:
        print(f"PASSED: Validation error caught successfully:")
        print(f"  --> {e}")


if __name__ == "__main__":
    print("=" * 65)
    print("BREAST CANCER CLASSIFICATION - PREDICTION PROGRAM (PREDICT.PY)")
    print("=" * 65)

    mode = "all"
    for arg in sys.argv[1:]:
        if arg in ["--sample=1", "--sample=malignant", "-m"]:
            mode = "sample1"
        elif arg in ["--sample=2", "--sample=benign", "-b"]:
            mode = "sample2"
        elif arg in ["--validate", "--test-validation", "-v"]:
            mode = "validate"

    if mode == "sample1":
        run_malignant_test()
    elif mode == "sample2":
        run_benign_test()
    elif mode == "validate":
        run_validation_test()
    else:
        run_malignant_test()
        run_benign_test()
        run_validation_test()

    print("\n" + "=" * 65)
    print("DISCLAIMER: This project is for educational and study purposes only.")
    print("It is not a substitute for professional clinical medical diagnosis.")
    print("=" * 65)
