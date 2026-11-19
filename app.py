# ==============================================================================
# Project: Breast Cancer Classification Using Machine Learning
# File: app.py
# Description: Clean and simple Streamlit web application for college project
#              demonstration. Uses the model trained on data/data.csv.
# ==============================================================================

import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="wide"
)

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS TO LOAD DATA AND SAVED MODEL
# ------------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    """Loads the dataset from data/data.csv."""
    path = os.path.join("data", "data.csv")
    if not os.path.exists(path):
        # Fallback if executed from subfolder
        path = os.path.join("..", "data", "data.csv")
    if not os.path.exists(path):
        st.error(f"Dataset file '{path}' not found.")
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_trained_model():
    """Loads the trained model artifacts saved by train_model.py."""
    path = "model.joblib"
    if not os.path.exists(path):
        # Fallback if executed from subfolder
        path = os.path.join("..", "model.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ------------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------------------
st.sidebar.title("Navigation")
menu_choice = st.sidebar.radio(
    "Go to Section:",
    [
        "1. Project Overview",
        "2. Dataset Information",
        "3. Exploratory Data Analysis",
        "4. Model Comparison & Evaluation",
        "5. Cancer Prediction"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project:** Breast Cancer Classification\n\n"
    "**Student:** College ML Project\n\n"
    "**Dataset:** Wisconsin Diagnostic Dataset"
)


# ------------------------------------------------------------------------------
# SECTION 1: PROJECT OVERVIEW
# ------------------------------------------------------------------------------
if menu_choice == "1. Project Overview":
    st.title("Breast Cancer Classification Using Machine Learning")

    st.write(
        "Welcome to the Breast Cancer Classification project. This machine learning application "
        "predicts whether a breast mass tumor is **Benign (non-cancerous)** or **Malignant (cancerous)** "
        "based on characteristics computed from digitized images of fine needle aspirates (FNA) of breast masses."
    )

    st.subheader("Project Objectives")
    st.write(
        "- Perform exploratory data analysis (EDA) to understand feature distributions and correlations.\n"
        "- Clean and preprocess the Wisconsin Breast Cancer dataset.\n"
        "- Train and compare three standard machine learning classifiers:\n"
        "  1. **Gaussian Naive Bayes** (probabilistic baseline studied in class)\n"
        "  2. **Logistic Regression** (linear classification model)\n"
        "  3. **Support Vector Machine (SVM)** (boundary-margin optimization)\n"
        "- Evaluate models using Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.\n"
        "- Provide an interactive prediction tool for demonstration."
    )

    st.warning(
        "⚠️ **Disclaimer:** This project is developed exclusively for academic and educational purposes "
        "as part of a college machine learning course. It is NOT intended or certified for actual clinical "
        "diagnosis or real-world medical decision-making."
    )


# ------------------------------------------------------------------------------
# SECTION 2: DATASET INFORMATION
# ------------------------------------------------------------------------------
elif menu_choice == "2. Dataset Information":
    st.title("Dataset Information")

    df = load_dataset()
    if df is not None:
        st.write(
            f"The dataset is loaded from `data/data.csv`. It contains **{df.shape[0]} rows** "
            f"and **{df.shape[1]} original columns**."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", df.shape[0])
        with col2:
            st.metric("Benign Samples (B)", int((df["diagnosis"] == "B").sum()))
        with col3:
            st.metric("Malignant Samples (M)", int((df["diagnosis"] == "M").sum()))

        st.subheader("First 5 Rows of the Dataset")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

        st.subheader("Data Cleaning Summary")
        st.write(
            "- **`id` Column:** Dropped because patient IDs are arbitrary unique numbers and not predictive biological features.\n"
            "- **`Unnamed: 32` Column:** Dropped because it contains 100% missing (NaN) values due to trailing commas in the CSV.\n"
            "- **Duplicate Rows:** 0 duplicate rows found in the dataset.\n"
            "- **Target Variable (`diagnosis`):** 'B' (Benign, 62.7%) and 'M' (Malignant, 37.3%)."
        )


# ------------------------------------------------------------------------------
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------------------------
elif menu_choice == "3. Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    st.write("Visualizations analyzing the distribution, relationships, and correlations of the features.")

    plot_option = st.selectbox(
        "Choose an EDA Visualization to Inspect:",
        [
            "Target Class Distribution",
            "Correlation Heatmap (Mean Features)",
            "Feature Histograms",
            "Feature Box Plots by Diagnosis",
            "Scatter Plot: Radius vs Texture"
        ]
    )

    if plot_option == "Target Class Distribution":
        st.subheader("Distribution of Diagnosis")
        if os.path.exists("plots/target_distribution.png"):
            st.image("plots/target_distribution.png", use_container_width=True)
        st.write(
            "**Observation:** The dataset contains 357 Benign (B) cases and 212 Malignant (M) cases. "
            "Because the classes are somewhat imbalanced (63% Benign, 37% Malignant), we use stratified splitting "
            "(`stratify=y`) during train-test split to maintain this exact ratio in both sets."
        )

    elif plot_option == "Correlation Heatmap (Mean Features)":
        st.subheader("Correlation Heatmap")
        if os.path.exists("plots/correlation_heatmap.png"):
            st.image("plots/correlation_heatmap.png", use_container_width=True)
        st.write(
            "**Observation:** Geometric features such as `radius_mean`, `perimeter_mean`, and `area_mean` "
            "show extremely high correlation (close to 1.0). This is expected because perimeter and area "
            "are mathematical functions of radius. This feature correlation violates the conditional independence "
            "assumption of Naive Bayes, which explains why Naive Bayes has a slightly lower recall."
        )

    elif plot_option == "Feature Histograms":
        st.subheader("Feature Distributions (Histograms)")
        if os.path.exists("plots/feature_histograms.png"):
            st.image("plots/feature_histograms.png", use_container_width=True)
        st.write(
            "**Observation:** For features like `radius_mean`, `perimeter_mean`, and `area_mean`, Malignant tumors "
            "(red) generally display significantly larger values than Benign tumors (green), making these strong "
            "separators for classification algorithms."
        )

    elif plot_option == "Feature Box Plots by Diagnosis":
        st.subheader("Box Plots Grouped by Diagnosis")
        if os.path.exists("plots/feature_boxplots.png"):
            st.image("plots/feature_boxplots.png", use_container_width=True)
        st.write(
            "**Observation:** The box plots clearly illustrate the differences in medians and quartiles. "
            "Malignant cases consistently show higher values and wider spread across all four selected features."
        )

    elif plot_option == "Scatter Plot: Radius vs Texture":
        st.subheader("Scatter Plot: Radius Mean vs Texture Mean")
        if os.path.exists("plots/scatter_plot.png"):
            st.image("plots/scatter_plot.png", use_container_width=True)
        st.write(
            "**Observation:** Plotting `radius_mean` against `texture_mean` shows a clear separation boundary "
            "between Benign and Malignant clusters, explaining why linear classifiers and SVM achieve high accuracy."
        )


# ------------------------------------------------------------------------------
# SECTION 4: MODEL COMPARISON & EVALUATION
# ------------------------------------------------------------------------------
elif menu_choice == "4. Model Comparison & Evaluation":
    st.title("Model Training & Evaluation Results")

    artifacts = load_trained_model()
    if artifacts is None:
        st.error("Model artifacts not found. Please run `py train_model.py` first.")
    else:
        st.subheader("Performance Comparison Table")
        comparison_df = artifacts["comparison_table"]
        st.dataframe(comparison_df.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}"
        }), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Accuracy Comparison")
            if os.path.exists("plots/model_comparison.png"):
                st.image("plots/model_comparison.png", use_container_width=True)

        with col2:
            st.subheader("Confusion Matrices")
            if os.path.exists("plots/confusion_matrices.png"):
                st.image("plots/confusion_matrices.png", use_container_width=True)

        st.subheader("Selected Model Explanation")
        st.write(
            f"**Selected Best Model:** **{artifacts['model_name']}**\n\n"
            f"- It achieved the highest test accuracy and the best F1-Score among all tested algorithms.\n"
            f"- It achieved high Recall on malignant tumors (identifying 39 out of 42 malignant cases in the test set).\n"
            f"- Gaussian Naive Bayes was trained as a baseline without scaling; however, due to correlated features, "
            f"it produced 7 false negatives.\n"
            f"- Logistic Regression and Support Vector Machine benefited significantly from `StandardScaler` normalization."
        )

        st.subheader("Why False Negatives Matter in Breast Cancer Detection")
        st.write(
            "In medical diagnostic systems, errors are not equal:\n\n"
            "- **False Positive (FP):** A benign tumor is predicted as malignant. The patient undergoes additional testing (e.g. biopsy), causing temporary anxiety, but no life is immediately lost.\n"
            "- **False Negative (FN):** A malignant cancer is predicted as benign. The patient is falsely reassured and cancer remains untreated, which can lead to disease progression and severe health consequences.\n\n"
            "Therefore, **high Recall (sensitivity)** for Malignant cases is the primary metric of concern."
        )


# ------------------------------------------------------------------------------
# SECTION 5: CANCER PREDICTION
# ------------------------------------------------------------------------------
elif menu_choice == "5. Cancer Prediction":
    st.title("Cancer Diagnosis Prediction")
    st.write("Enter the 30 cell nucleus measurements or load real patient samples to test the trained model.")

    artifacts = load_trained_model()
    if artifacts is None:
        st.error("Model file 'model.joblib' not found. Please run `py train_model.py` first.")
    else:
        feature_names = artifacts["feature_names"]
        model = artifacts["model"]
        scaler = artifacts["scaler"]
        model_name = artifacts["model_name"]

        st.info(f"Loaded Trained Model: **{model_name}**")

        # ----------------------------------------------------------------------
        # PRESET SAMPLES FROM ACTUAL DATASET
        # ----------------------------------------------------------------------
        # Sample 1: Row 0 (Confirmed Malignant in data/data.csv)
        sample_malignant_row0 = {
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

        # Sample 2: Row 19 (Confirmed Benign in data/data.csv)
        sample_benign_row19 = {
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

        # Sample 3: Row 20 (Confirmed Benign in data/data.csv)
        sample_benign_row20 = {
            'radius_mean': 13.08, 'texture_mean': 15.71, 'perimeter_mean': 85.63,
            'area_mean': 520.0, 'smoothness_mean': 0.1075, 'compactness_mean': 0.1270,
            'concavity_mean': 0.04568, 'concave points_mean': 0.0311, 'symmetry_mean': 0.1967,
            'fractal_dimension_mean': 0.06811, 'radius_se': 0.1852, 'texture_se': 0.7477,
            'perimeter_se': 1.383, 'area_se': 14.67, 'smoothness_se': 0.004097,
            'compactness_se': 0.01898, 'concavity_se': 0.01698, 'concave points_se': 0.00649,
            'symmetry_se': 0.01678, 'fractal_dimension_se': 0.002425, 'radius_worst': 14.50,
            'texture_worst': 20.49, 'perimeter_worst': 96.09, 'area_worst': 630.5,
            'smoothness_worst': 0.1312, 'compactness_worst': 0.2776, 'concavity_worst': 0.1890,
            'concave points_worst': 0.07283, 'symmetry_worst': 0.3184,
            'fractal_dimension_worst': 0.08183
        }

        # Realistic Benign Baseline (Exact Medians of all 357 Benign tumors in data.csv)
        benign_medians = {
            'radius_mean': 12.20, 'texture_mean': 17.39, 'perimeter_mean': 78.18,
            'area_mean': 458.4, 'smoothness_mean': 0.09076, 'compactness_mean': 0.07529,
            'concavity_mean': 0.03709, 'concave points_mean': 0.02344, 'symmetry_mean': 0.1714,
            'fractal_dimension_mean': 0.06154, 'radius_se': 0.2575, 'texture_se': 1.1080,
            'perimeter_se': 1.851, 'area_se': 19.63, 'smoothness_se': 0.00653,
            'compactness_se': 0.01631, 'concavity_se': 0.01840, 'concave points_se': 0.00906,
            'symmetry_se': 0.01909, 'fractal_dimension_se': 0.00281, 'radius_worst': 13.35,
            'texture_worst': 22.82, 'perimeter_worst': 86.92, 'area_worst': 547.4,
            'smoothness_worst': 0.1254, 'compactness_worst': 0.1698, 'concavity_worst': 0.1412,
            'concave points_worst': 0.07431, 'symmetry_worst': 0.2687,
            'fractal_dimension_worst': 0.07712
        }

        # Helper function to reliably update session state input values
        def populate_inputs(sample_dict, sample_name="Custom"):
            st.session_state["loaded_sample_name"] = sample_name
            for col in feature_names:
                st.session_state[f"field_{col}"] = float(sample_dict.get(col, benign_medians.get(col, 0.0)))

        # Initialize session state with benign medians on first visit
        if "field_radius_mean" not in st.session_state:
            populate_inputs(benign_medians, "Benign Baseline Medians")

        # ----------------------------------------------------------------------
        # QUICK VIVA DEMONSTRATION BUTTONS
        # ----------------------------------------------------------------------
        st.subheader("1. Quick Sample Loaders (Recommended for Viva Demo)")
        st.write("Click any button to instantly load confirmed patient data from `data/data.csv`:")

        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

        with btn_col1:
            if st.button("🚨 Load Sample 1 (Malignant Row 0)"):
                populate_inputs(sample_malignant_row0, "Row 0: Malignant Case")
                st.rerun()

        with btn_col2:
            if st.button("✅ Load Sample 2 (Benign Row 19)"):
                populate_inputs(sample_benign_row19, "Row 19: Benign Case")
                st.rerun()

        with btn_col3:
            if st.button("✅ Load Sample 3 (Benign Row 20)"):
                populate_inputs(sample_benign_row20, "Row 20: Benign Case")
                st.rerun()

        with btn_col4:
            if st.button("🔄 Reset (Benign Medians)"):
                populate_inputs(benign_medians, "Benign Baseline Medians")
                st.rerun()

        active_loaded = st.session_state.get("loaded_sample_name", "Benign Baseline Medians")
        if "Malignant" in active_loaded:
            st.markdown(f"**Currently Loaded Sample:** :red[{active_loaded}] — *(Expected Ground Truth: Malignant / Cancerous)*")
        else:
            st.markdown(f"**Currently Loaded Sample:** :green[{active_loaded}] — *(Expected Ground Truth: Benign / Non-Cancerous)*")

        # ----------------------------------------------------------------------
        # FEATURE INPUT FIELDS (ORGANIZED IN 3 TABS)
        # ----------------------------------------------------------------------
        st.subheader("2. Review or Edit 30 Cell Nucleus Features")

        tab1, tab2, tab3 = st.tabs(["Mean Features (10)", "Standard Error (SE) Features (10)", "Worst Features (10)"])

        user_inputs = {}

        with tab1:
            st.caption("Mean physical measurements of cell nuclei:")
            mean_cols = [c for c in feature_names if "_mean" in c]
            cols = st.columns(2)
            for i, col in enumerate(mean_cols):
                with cols[i % 2]:
                    user_inputs[col] = st.number_input(
                        f"{col}",
                        value=float(st.session_state.get(f"field_{col}", benign_medians[col])),
                        format="%.5f",
                        key=f"field_{col}"
                    )

        with tab2:
            st.caption("Standard error of physical measurements:")
            se_cols = [c for c in feature_names if "_se" in c]
            cols = st.columns(2)
            for i, col in enumerate(se_cols):
                with cols[i % 2]:
                    user_inputs[col] = st.number_input(
                        f"{col}",
                        value=float(st.session_state.get(f"field_{col}", benign_medians[col])),
                        format="%.5f",
                        key=f"field_{col}"
                    )

        with tab3:
            st.caption("Worst (largest) physical measurements:")
            worst_cols = [c for c in feature_names if "_worst" in c]
            cols = st.columns(2)
            for i, col in enumerate(worst_cols):
                with cols[i % 2]:
                    user_inputs[col] = st.number_input(
                        f"{col}",
                        value=float(st.session_state.get(f"field_{col}", benign_medians[col])),
                        format="%.5f",
                        key=f"field_{col}"
                    )

        # ----------------------------------------------------------------------
        # PREDICTION EXECUTION & RESULTS
        # ----------------------------------------------------------------------
        st.markdown("---")
        st.subheader("3. Run Prediction")

        if st.button("🔮 Predict Tumor Diagnosis", type="primary"):
            # Arrange input features in exact column order expected by trained model
            input_df = pd.DataFrame([user_inputs])[feature_names]

            # Scale if model is Logistic Regression or Support Vector Machine
            if model_name in ["Logistic Regression", "Support Vector Machine"]:
                input_scaled = scaler.transform(input_df)
            else:
                input_scaled = input_df

            # Predict class and decision score
            prediction = model.predict(input_scaled)[0]

            decision_score = None
            if hasattr(model, "decision_function"):
                decision_score = float(model.decision_function(input_scaled)[0])

            st.subheader("Prediction Result:")

            if prediction == "M":
                st.error(
                    "🚨 **Prediction: Malignant (Cancerous)**\n\n"
                    "The cell nucleus measurements exhibit patterns characteristic of a malignant breast tumor "
                    "(e.g., significantly enlarged radius, perimeter, and elevated concavity)."
                )
            else:
                st.success(
                    "✅ **Prediction: Benign (Non-Cancerous)**\n\n"
                    "The cell nucleus measurements exhibit patterns characteristic of a benign breast mass "
                    "(physical dimensions and texture values are within normal non-cancerous ranges)."
                )

            # Mathematical transparency for viva presentation
            if decision_score is not None:
                st.caption(
                    f"**SVM Decision Score:** `{decision_score:+.4f}` "
                    f"*(Negative score < 0 indicates Benign region; Positive score > 0 indicates Malignant region)*"
                )

            st.info(
                f"Prediction computed by **{model_name}** using `StandardScaler` feature normalization fitted on `data/data.csv`."
            )

            st.warning(
                "⚠️ **Educational Project Notice:** This prediction is generated by an academic machine learning "
                "model and is strictly for educational demonstration. It must not be used for medical decisions."
            )
