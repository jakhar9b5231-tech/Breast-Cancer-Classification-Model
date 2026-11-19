 ==============================================================================
# Project: Breast Cancer Classification Using Machine Learning
# File: app.py
    """Loads the dataset from data/data.csv."""
    path = os.path.join("data", "data.csv")
    if not os.path.exists(path):
        # Fallback if executed from subfolder
        path = os.path.join("..", "data", "data.csv")
    if not os.path.exists(path):
        st.error(f"Dataset file '{path}' not found.")
        return None
    return pd.read_csv(path)
    """Loads the trained model artifacts saved by train_model.py."""
    path = "model.joblib"
    if not os.path.exists(path):
        # Fallback if executed from subfolder
        path = os.path.join("..", "model.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)
# ------------------------------------------------------------------------------
elif menu_choice == "5. Cancer Prediction":
    st.title("Cancer Diagnosis Prediction")
    st.write("Enter the 30 cell nucleus measurements or load sample cases to test the trained model.")
    st.write("Enter the 30 cell nucleus measurements or load real patient samples to test the trained model.")
    artifacts = load_trained_model()
    if artifacts is None:
        st.info(f"Loaded Trained Model: **{model_name}**")
        # Preset test samples taken directly from data/data.csv
        sample_malignant = {
        # ----------------------------------------------------------------------
        # PRESET SAMPLES FROM ACTUAL DATASET
        # ----------------------------------------------------------------------
        # Sample 1: Row 0 (Confirmed Malignant in data/data.csv)
        sample_malignant_row0 = {
            'radius_mean': 17.99, 'texture_mean': 10.38, 'perimeter_mean': 122.8,
            'area_mean': 1001.0, 'smoothness_mean': 0.1184, 'compactness_mean': 0.2776,
            'concavity_mean': 0.3001, 'concave points_mean': 0.1471, 'symmetry_mean': 0.2419,
            'fractal_dimension_worst': 0.1189
        }
        sample_benign = {
