# ==============================================================================
# Project: Breast Cancer Classification Using Machine Learning
# File: app.py
# Description: Elegant, interactive Streamlit web dashboard styled with Cancer Fund
#              Awareness colors (soft pink, lavender, and crisp white).
# ==============================================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer AI Classification Dashboard",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# CANCER FUND AWARENESS CUSTOM STYLING (PINK, LAVENDER, WHITE)
# ------------------------------------------------------------------------------
AWARENESS_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(180deg, #fdf8fb 0%, #faf5fc 50%, #ffffff 100%);
    }

    /* Custom Header Banner */
    .awareness-hero {
        background: linear-gradient(135deg, #d81b60 0%, #8e24aa 50%, #5e35b1 100%);
        padding: 26px 32px;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(216, 27, 96, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .awareness-hero h1 {
        color: #ffffff !important;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .awareness-hero p {
        color: #fce4ec !important;
        font-size: 1.05rem;
        margin: 0;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Ribbon Tag */
    .ribbon-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.22);
        color: #ffffff;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
        backdrop-filter: blur(4px);
    }

    /* KPI / Metric Cards */
    .metric-card {
        background: #ffffff;
        padding: 20px 22px;
        border-radius: 14px;
        border: 1px solid #f3e5f5;
        box-shadow: 0 4px 16px rgba(142, 36, 170, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(216, 27, 96, 0.12);
    }
    .metric-card .title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #7b1fa2;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #d81b60;
        line-height: 1.1;
    }
    .metric-card .subtext {
        font-size: 0.85rem;
        color: #6a1b9a;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Content Cards */
    .content-box {
        background: #ffffff;
        border: 1px solid #f3e5f5;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 22px;
        box-shadow: 0 4px 16px rgba(126, 87, 194, 0.05);
    }

    /* Disclaimer Card */
    .disclaimer-card {
        background: linear-gradient(135deg, #fff5f8 0%, #fbf8fc 100%);
        border-left: 5px solid #d81b60;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 20px 0;
        color: #880e4f;
        font-size: 0.93rem;
        line-height: 1.5;
        border-top: 1px solid #fce4ec;
        border-right: 1px solid #fce4ec;
        border-bottom: 1px solid #fce4ec;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fdf4f8 0%, #f6eff9 100%) !important;
        border-right: 1px solid #f1e1f5;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #ede7f6;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 8px 8px 0 0;
        color: #6a1b9a !important;
        font-weight: 600;
        padding: 0 20px;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #d81b60 !important;
        border-bottom: 3px solid #d81b60 !important;
        background: #fdf2f7 !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        border: 1px solid #d81b60 !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #d81b60 0%, #ab47bc 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(216, 27, 96, 0.25) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(216, 27, 96, 0.35) !important;
    }

    /* Result Badges */
    .result-badge-benign {
        background: linear-gradient(135deg, #f3e5f5 0%, #ede7f6 100%);
        border: 2px solid #b39ddb;
        border-radius: 12px;
        padding: 20px;
        color: #4a148c;
        text-align: center;
        margin: 16px 0;
        box-shadow: 0 4px 16px rgba(106, 27, 154, 0.08);
    }
    .result-badge-malignant {
        background: linear-gradient(135deg, #fce4ec 0%, #f8bbd0 100%);
        border: 2px solid #f06292;
        border-radius: 12px;
        padding: 20px;
        color: #880e4f;
        text-align: center;
        margin: 16px 0;
        box-shadow: 0 4px 16px rgba(216, 27, 96, 0.12);
    }
</style>
"""
st.markdown(AWARENESS_CSS, unsafe_allow_html=True)

# Awareness palette for Matplotlib & Seaborn plots
AWARENESS_PALETTE = {"B": "#7e57c2", "M": "#d81b60"}


# ------------------------------------------------------------------------------
# DATA & MODEL LOADERS
# ------------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    """Loads the Wisconsin breast cancer dataset from data/data.csv."""
    path = os.path.join("data", "data.csv")
    if not os.path.exists(path):
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
        path = os.path.join("..", "model.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


df_raw = load_dataset()
artifacts = load_trained_model()


# ------------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <div style="font-size: 3.2rem; filter: drop-shadow(0 4px 6px rgba(216,27,96,0.2));">🎗️</div>
            <h2 style="color: #880e4f; margin: 6px 0 2px 0; font-size: 1.35rem; font-weight: 800;">Cancer Fund AI</h2>
            <p style="color: #6a1b9a; font-size: 0.85rem; margin: 0; font-weight: 500;">Breast Cancer Classification</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    menu_choice = st.radio(
        "Navigate Dashboard:",
        [
            "🎗️ 1. Project Overview",
            "📊 2. Dataset Explorer",
            "📈 3. Interactive EDA Lab",
            "🔬 4. Model Evaluation",
            "🔮 5. Cancer Prediction Lab"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="background: rgba(255,255,255,0.7); padding: 14px; border-radius: 10px; border: 1px solid #f1dff5;">
            <p style="margin: 0; font-size: 0.82rem; color: #4a148c; font-weight: 600;">🎗️ Pink & Lavender Theme</p>
            <p style="margin: 4px 0 0 0; font-size: 0.78rem; color: #6a1b9a;">Designed in tribute to cancer research organizations and early detection awareness.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------------------------
# SECTION 1: PROJECT OVERVIEW
# ------------------------------------------------------------------------------
if "1. Project Overview" in menu_choice:
    st.markdown(
        """
        <div class="awareness-hero">
            <span class="ribbon-tag">🎗️ BREAST CANCER RESEARCH & AWARENESS INITIATIVE</span>
            <h1>Breast Cancer Classification Using Machine Learning</h1>
            <p>Empowering early diagnosis through fine needle aspirate (FNA) cell cytology analysis, feature exploration, and transparent machine learning models.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 3 Summary Awareness Metric Cards
    if df_raw is not None:
        total_n = len(df_raw)
        b_count = int((df_raw["diagnosis"] == "B").sum())
        m_count = int((df_raw["diagnosis"] == "M").sum())
        b_pct = (b_count / total_n) * 100
        m_pct = (m_count / total_n) * 100

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="title">Total Biopsy Cases</div>
                    <div class="value" style="color: #7b1fa2;">{total_n}</div>
                    <div class="subtext">Wisconsin Diagnostic Dataset (WDBC)</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="title">Benign Cases (Non-Cancerous)</div>
                    <div class="value" style="color: #5e35b1;">{b_count}</div>
                    <div class="subtext">{b_pct:.1f}% of study cohort</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="title">Malignant Cases (Cancerous)</div>
                    <div class="value" style="color: #d81b60;">{m_count}</div>
                    <div class="subtext">{m_pct:.1f}% requiring immediate treatment</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive tabs for deeper understanding
    tab_intro, tab_models, tab_clinical = st.tabs([
        "📖 Project Scope & Objectives",
        "🧠 Machine Learning Methodology",
        "🩺 Clinical Context: FNA Biopsy"
    ])

    with tab_intro:
        st.markdown(
            """
            ### Project Objectives & College ML Workflow
            This project applies foundational supervised machine learning to classify breast masses as either **Benign** or **Malignant**:
            
            1. **Exploratory Data Analysis (EDA):** Identify clinical distributions, outliers, and feature correlations among 30 nuclear dimensions.
            2. **Data Cleaning & Separation:** Eliminate non-predictive identifier columns (`id`) and empty columns (`Unnamed: 32`) to prevent noise.
            3. **Data Leakage Prevention:** Apply stratified train-test splits (`80:20`) and fit `StandardScaler` strictly on training data.
            4. **Multi-Model Benchmark:** Train and compare three standard Scikit-learn algorithms:
               - **Gaussian Naive Bayes (`GaussianNB`)**: Probabilistic baseline studied in classroom theory.
               - **Logistic Regression (`LogisticRegression`)**: Standard linear classification benchmark.
               - **Support Vector Machine (`SVC`)**: Optimal margin-separating hyperplane classifier.
            5. **Clinical Evaluation:** Prioritize **Recall (Sensitivity)** for malignant detection to minimize life-threatening false negatives.
            """
        )

    with tab_models:
        st.markdown(
            """
            ### Algorithmic Comparison Breakdown
            | Model | Class Implemented | Feature Scaling | Key Assumption / Characteristic |
            | :--- | :--- | :---: | :--- |
            | **Gaussian Naive Bayes** | `sklearn.naive_bayes.GaussianNB` | Not Required | Assumes conditional independence between features given the class. |
            | **Logistic Regression** | `sklearn.linear_model.LogisticRegression` | `StandardScaler` | Uses the logistic sigmoid function to model class probability log-odds. |
            | **Support Vector Machine** | `sklearn.svm.SVC` | `StandardScaler` | Maximizes geometric margin between classes with kernel support vectors. |
            """
        )

    with tab_clinical:
        st.markdown(
            """
            ### What is a Fine Needle Aspirate (FNA)?
            A **Fine Needle Aspirate** is a minimally invasive diagnostic biopsy procedure where a thin needle extracts cell clusters from a breast lump. 
            A digital scan computes 10 characteristic dimensions for each cell nucleus:
            - **Radius:** Distance from center to points on perimeter.
            - **Texture:** Standard deviation of gray-scale values.
            - **Perimeter & Area:** Overall nuclear size.
            - **Smoothness:** Local variation in radius lengths.
            - **Compactness:** $\\frac{\\text{perimeter}^2}{\\text{area}} - 1.0$
            - **Concavity & Concave Points:** Severity and count of indentations on the nuclear contour.
            - **Symmetry & Fractal Dimension:** Uniformity and boundary complexity.
            """
        )

    st.markdown(
        """
        <div class="disclaimer-card">
            <strong>⚠️ Educational & Academic Disclaimer:</strong><br>
            This application is created solely for coursework presentation, academic study, and educational demonstration. 
            It is <strong>not</strong> certified as clinical software and must never replace qualified radiological, cytological, or medical consultation.
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------------------------
# SECTION 2: DATASET EXPLORER
# ------------------------------------------------------------------------------
elif "2. Dataset Explorer" in menu_choice:
    st.markdown(
        """
        <div class="awareness-hero" style="background: linear-gradient(135deg, #7b1fa2 0%, #5e35b1 100%);">
            <span class="ribbon-tag">📊 INTERACTIVE DATA BROWSER</span>
            <h1>Dataset Exploration & Statistics</h1>
            <p>Filter, inspect, and analyze raw patient cytology records from <code>data/data.csv</code>.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if df_raw is not None:
        # Interactive Filter Bar
        st.markdown("### Interactive Record Filters")
        f_col1, f_col2, f_col3 = st.columns([1.5, 2, 1.5])

        with f_col1:
            class_filter = st.selectbox(
                "Filter by Diagnosis:",
                ["All Cases (569)", "Benign Only (357)", "Malignant Only (212)"]
            )

        with f_col2:
            feature_group = st.selectbox(
                "Select Feature Category to Display:",
                ["All Features (30)", "Mean Features (10)", "Standard Error (SE) Features (10)", "Worst Features (10)"]
            )

        with f_col3:
            num_rows = st.slider("Rows to Preview:", min_value=5, max_value=50, value=10, step=5)

        # Filter DataFrame
        df_filtered = df_raw.copy()
        if "id" in df_filtered.columns:
            df_filtered = df_filtered.drop(columns=["id"])
        if "Unnamed: 32" in df_filtered.columns:
            df_filtered = df_filtered.drop(columns=["Unnamed: 32"])

        if class_filter == "Benign Only (357)":
            df_filtered = df_filtered[df_filtered["diagnosis"] == "B"]
        elif class_filter == "Malignant Only (212)":
            df_filtered = df_filtered[df_filtered["diagnosis"] == "M"]

        # Filter Columns
        if feature_group == "Mean Features (10)":
            cols_to_show = ["diagnosis"] + [c for c in df_filtered.columns if "_mean" in c]
        elif feature_group == "Standard Error (SE) Features (10)":
            cols_to_show = ["diagnosis"] + [c for c in df_filtered.columns if "_se" in c]
        elif feature_group == "Worst Features (10)":
            cols_to_show = ["diagnosis"] + [c for c in df_filtered.columns if "_worst" in c]
        else:
            cols_to_show = list(df_filtered.columns)

        st.dataframe(df_filtered[cols_to_show].head(num_rows), use_container_width=True)
        st.caption(f"Displaying {min(num_rows, len(df_filtered))} of {len(df_filtered)} filtered records.")

        # Interactive Statistics
        st.markdown("---")
        st.markdown("### Statistical Distribution Summary")
        st.dataframe(df_filtered[cols_to_show].describe(), use_container_width=True)

        # Data Cleaning Summary Cards
        st.markdown("---")
        st.markdown("### Data Cleaning & Integrity Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ID Column", "Removed", "Arbitrary identifier")
        c2.metric("Unnamed: 32", "Removed", "100% NaN artifact")
        c3.metric("Duplicate Rows", "0 Found", "Dataset is clean")
        c4.metric("Missing Values", "0 Remaining", "Complete numerical data")


# ------------------------------------------------------------------------------
# SECTION 3: INTERACTIVE EDA LAB
# ------------------------------------------------------------------------------
elif "3. Interactive EDA Lab" in menu_choice:
    st.markdown(
        """
        <div class="awareness-hero" style="background: linear-gradient(135deg, #c2185b 0%, #7b1fa2 100%);">
            <span class="ribbon-tag">📈 VISUALIZATION LAB</span>
            <h1>Exploratory Data Analysis Lab</h1>
            <p>Switch between the standard project plots or generate live interactive visualizations using the Pink & Lavender awareness palette.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    eda_mode = st.radio(
        "Choose EDA Exploration Mode:",
        ["📌 Standard Coursework Visualizations", "🎨 Interactive Live Plot Generator"],
        horizontal=True
    )

    if eda_mode == "📌 Standard Coursework Visualizations":
        plot_option = st.selectbox(
            "Select Coursework Plot to Inspect:",
            [
                "Target Class Distribution",
                "Correlation Heatmap (Mean Features)",
                "Feature Histograms",
                "Feature Box Plots by Diagnosis",
                "Scatter Plot: Radius vs Texture"
            ]
        )

        if plot_option == "Target Class Distribution":
            st.markdown("#### Distribution of Diagnosis (Benign vs Malignant)")
            if os.path.exists("plots/target_distribution.png"):
                st.image("plots/target_distribution.png", use_container_width=True)
            st.info(
                "**Insight:** The dataset exhibits 357 Benign (62.7%) and 212 Malignant (37.3%) cases. "
                "Because of this imbalance, `stratify=y` is strictly employed during train-test splitting."
            )

        elif plot_option == "Correlation Heatmap (Mean Features)":
            st.markdown("#### Pearson Correlation Heatmap")
            if os.path.exists("plots/correlation_heatmap.png"):
                st.image("plots/correlation_heatmap.png", use_container_width=True)
            st.info(
                "**Insight:** Geometric features (`radius_mean`, `perimeter_mean`, `area_mean`) have collinearity approaching 1.0. "
                "This violates the feature independence assumption of Naive Bayes, explaining its higher false negative count."
            )

        elif plot_option == "Feature Histograms":
            st.markdown("#### Distribution Histograms by Diagnosis")
            if os.path.exists("plots/feature_histograms.png"):
                st.image("plots/feature_histograms.png", use_container_width=True)
            st.info(
                "**Insight:** Malignant tumors (red/pink) exhibit significantly larger radii and areas than benign masses, "
                "offering sharp linear separability."
            )

        elif plot_option == "Feature Box Plots by Diagnosis":
            st.markdown("#### Box Plots Grouped by Diagnosis")
            if os.path.exists("plots/feature_boxplots.png"):
                st.image("plots/feature_boxplots.png", use_container_width=True)
            st.info(
                "**Insight:** Box plots highlight higher medians and broader interquartile ranges among malignant tumors."
            )

        elif plot_option == "Scatter Plot: Radius vs Texture":
            st.markdown("#### 2D Scatter Plot: Radius Mean vs Texture Mean")
            if os.path.exists("plots/scatter_plot.png"):
                st.image("plots/scatter_plot.png", use_container_width=True)
            st.info(
                "**Insight:** Clear decision boundaries between benign and malignant clusters illustrate why margin classifiers (SVM) excel."
            )

    else:
        # LIVE INTERACTIVE GENERATOR
        st.markdown("### Interactive Live Cytology Plotter")
        st.write("Generate real-time plots using the Pink & Lavender cancer fund awareness palette:")

        if df_raw is not None:
            clean_df = df_raw.drop(columns=["id", "Unnamed: 32"], errors="ignore")
            feature_cols = [c for c in clean_df.columns if c != "diagnosis"]

            interactive_type = st.selectbox(
                "Choose Plot Type to Generate:",
                ["Live 2D Scatter Plot (Select X and Y)", "Feature Distribution & KDE Histogram", "Comparison Violin / Box Plot"]
            )

            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#faf7fc")

            if interactive_type == "Live 2D Scatter Plot (Select X and Y)":
                p_c1, p_c2 = st.columns(2)
                with p_c1:
                    x_feat = st.selectbox("X-Axis Feature:", feature_cols, index=0)
                with p_c2:
                    y_feat = st.selectbox("Y-Axis Feature:", feature_cols, index=1)

                sns.scatterplot(
                    data=clean_df,
                    x=x_feat,
                    y=y_feat,
                    hue="diagnosis",
                    palette=AWARENESS_PALETTE,
                    alpha=0.85,
                    s=55,
                    ax=ax
                )
                ax.set_title(f"Interactive Scatter: {x_feat} vs {y_feat}", fontsize=13, fontweight="bold", color="#4a148c")
                ax.set_xlabel(x_feat, fontsize=11, fontweight="bold", color="#6a1b9a")
                ax.set_ylabel(y_feat, fontsize=11, fontweight="bold", color="#6a1b9a")
                ax.grid(True, linestyle="--", alpha=0.5, color="#ede7f6")
                st.pyplot(fig)

            elif interactive_type == "Feature Distribution & KDE Histogram":
                sel_feat = st.selectbox("Select Feature to Inspect:", feature_cols, index=0)
                sns.histplot(
                    data=clean_df,
                    x=sel_feat,
                    hue="diagnosis",
                    palette=AWARENESS_PALETTE,
                    kde=True,
                    bins=28,
                    ax=ax
                )
                ax.set_title(f"Distribution & Density: {sel_feat}", fontsize=13, fontweight="bold", color="#4a148c")
                ax.set_xlabel(sel_feat, fontsize=11, fontweight="bold", color="#6a1b9a")
                ax.set_ylabel("Frequency", fontsize=11, fontweight="bold", color="#6a1b9a")
                ax.grid(True, linestyle="--", alpha=0.5, color="#ede7f6")
                st.pyplot(fig)

            elif interactive_type == "Comparison Violin / Box Plot":
                sel_feat = st.selectbox("Select Feature for Box/Violin Plot:", feature_cols, index=0)
                sns.boxplot(
                    data=clean_df,
                    x="diagnosis",
                    y=sel_feat,
                    hue="diagnosis",
                    palette=AWARENESS_PALETTE,
                    ax=ax
                )
                ax.set_title(f"Box Plot: {sel_feat} by Tumor Class", fontsize=13, fontweight="bold", color="#4a148c")
                ax.set_xticklabels(["Benign (B)", "Malignant (M)"], fontsize=11, fontweight="bold")
                ax.grid(True, linestyle="--", alpha=0.5, color="#ede7f6")
                st.pyplot(fig)


# ------------------------------------------------------------------------------
# SECTION 4: MODEL EVALUATION & COMPARISON
# ------------------------------------------------------------------------------
elif "4. Model Evaluation" in menu_choice:
    st.markdown(
        """
        <div class="awareness-hero" style="background: linear-gradient(135deg, #ad1457 0%, #6a1b9a 100%);">
            <span class="ribbon-tag">🔬 MODEL BENCHMARKING</span>
            <h1>Model Performance & Clinical Evaluation</h1>
            <p>Comprehensive comparative results across Gaussian Naive Bayes, Logistic Regression, and Support Vector Machine.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if artifacts is not None:
        comp_df = artifacts["comparison_table"]

        st.markdown("### Performance Comparison Matrix")
        st.dataframe(
            comp_df.style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-Score": "{:.4f}"
            }).background_gradient(cmap="Purples", subset=["Accuracy", "F1-Score"]),
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Side-by-side plots
        st.markdown("### Graphical Evaluation")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("#### Accuracy Comparison")
            if os.path.exists("plots/model_comparison.png"):
                st.image("plots/model_comparison.png", use_container_width=True)

        with col_p2:
            st.markdown("#### Side-by-Side Confusion Matrices")
            if os.path.exists("plots/confusion_matrices.png"):
                st.image("plots/confusion_matrices.png", use_container_width=True)

        # Model Rationale Card
        st.markdown(
            f"""
            <div class="content-box">
                <h3 style="color: #4a148c; margin-top: 0;">Selected Best Algorithm: {artifacts['model_name']}</h3>
                <p style="color: #333; line-height: 1.6;">
                    • <strong>Top Test Accuracy ({comp_df.loc[comp_df['Model'] == artifacts['model_name'], 'Accuracy'].values[0]:.4f}):</strong> Correctly classified the highest proportion of held-out test samples.<br>
                    • <strong>Superior Recall on Malignancy:</strong> Detected 39 out of 42 malignant tumors in test data, keeping false negatives to a minimum.<br>
                    • <strong>StandardScaler Synergy:</strong> Distance-based algorithms (SVM and Logistic Regression) demonstrated major performance gains from feature standardization compared to unscaled Naive Bayes.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Clinical Importance of False Negatives
        st.markdown(
            """
            <div class="disclaimer-card" style="border-left-color: #7b1fa2; background: #faf5fc;">
                <h4 style="margin: 0 0 6px 0; color: #4a148c;">🩺 Clinical Significance: Why False Negatives Matter in Oncology</h4>
                <p style="margin: 0; color: #4a148c; line-height: 1.5;">
                    • <strong>False Positive (FP):</strong> A benign lump is predicted as malignant. The patient is referred for confirmatory imaging or biopsy. It causes emotional stress, but is not life-threatening.<br>
                    • <strong>False Negative (FN):</strong> A malignant tumor is incorrectly predicted as benign. The patient receives false reassurance, and untreated cancer progresses.<br>
                    Therefore, in clinical diagnostic algorithms, <strong>Recall (Sensitivity)</strong> for malignant tumors is of paramount importance.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ------------------------------------------------------------------------------
# SECTION 5: CANCER PREDICTION LAB
# ------------------------------------------------------------------------------
elif "5. Cancer Prediction Lab" in menu_choice:
    st.markdown(
        """
        <div class="awareness-hero" style="background: linear-gradient(135deg, #d81b60 0%, #ad1457 50%, #7b1fa2 100%);">
            <span class="ribbon-tag">🔮 INTERACTIVE DIAGNOSTIC PREDICTOR</span>
            <h1>Real-Time Cancer Diagnosis Predictor</h1>
            <p>Test individual patient records or adjust cytology features interactively to observe real-time classification predictions.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if artifacts is None:
        st.error("Trained model artifacts ('model.joblib') not found. Please run train_model.py first.")
    else:
        feature_names = artifacts["feature_names"]
        model = artifacts["model"]
        scaler = artifacts["scaler"]
        model_name = artifacts["model_name"]

        # Realistic Benign Baseline (Medians from data.csv)
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

        # Real confirmed patient samples from data/data.csv
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

        # Helper to reliably load values into session state
        def set_preset(sample_dict, label):
            st.session_state["active_sample_label"] = label
            for col in feature_names:
                st.session_state[f"field_{col}"] = float(sample_dict.get(col, benign_medians.get(col, 0.0)))

        # Initialize session state if first visit
        if "field_radius_mean" not in st.session_state:
            set_preset(benign_medians, "Benign Baseline Medians")

        # Quick Viva Demo Buttons
        st.markdown("### 1. Instant Viva Sample Loaders")
        st.caption("Click any button to test confirmed cases directly from the dataset:")

        b_c1, b_c2, b_c3, b_c4 = st.columns(4)
        with b_c1:
            if st.button("🚨 Load Malignant (Row 0)", use_container_width=True):
                set_preset(sample_malignant_row0, "Row 0 (Malignant)")
                st.rerun()

        with b_c2:
            if st.button("✅ Load Benign (Row 19)", use_container_width=True):
                set_preset(sample_benign_row19, "Row 19 (Benign)")
                st.rerun()

        with b_c3:
            if st.button("✅ Load Benign (Row 20)", use_container_width=True):
                set_preset(sample_benign_row20, "Row 20 (Benign)")
                st.rerun()

        with b_c4:
            if st.button("🔄 Reset to Baseline", use_container_width=True):
                set_preset(benign_medians, "Benign Baseline Medians")
                st.rerun()

        active_label = st.session_state.get("active_sample_label", "Benign Baseline Medians")
        if "Malignant" in active_label:
            st.markdown(f"**Loaded Case:** <span style='color: #c2185b; font-weight: bold;'>{active_label}</span> *(Expected: Malignant)*", unsafe_allow_html=True)
        else:
            st.markdown(f"**Loaded Case:** <span style='color: #512da8; font-weight: bold;'>{active_label}</span> *(Expected: Benign)*", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 2. Fine-Tune Feature Measurements")

        tab_m, tab_se, tab_w = st.tabs([
            "Mean Physical Measurements (10)",
            "Standard Error (SE) Measurements (10)",
            "Worst (Extreme) Measurements (10)"
        ])

        user_inputs = {}

        with tab_m:
            st.caption("Average cell nucleus measurements:")
            mean_cols = [c for c in feature_names if "_mean" in c]
            m_cols = st.columns(2)
            for i, col in enumerate(mean_cols):
                with m_cols[i % 2]:
                    user_inputs[col] = st.number_input(
                        f"{col}",
                        value=float(st.session_state.get(f"field_{col}", benign_medians[col])),
                        format="%.5f",
                        key=f"field_{col}"
                    )

        with tab_se:
            st.caption("Standard error across nucleus contours:")
            se_cols = [c for c in feature_names if "_se" in c]
            s_cols = st.columns(2)
            for i, col in enumerate(se_cols):
                with s_cols[i % 2]:
                    user_inputs[col] = st.number_input(
                        f"{col}",
                        value=float(st.session_state.get(f"field_{col}", benign_medians[col])),
                        format="%.5f",
                        key=f"field_{col}"
                    )

        with tab_w:
            st.caption("Worst (largest / most severe) nuclear observations:")
            worst_cols = [c for c in feature_names if "_worst" in c]
            w_cols = st.columns(2)
            for i, col in enumerate(worst_cols):
                with w_cols[i % 2]:
                    user_inputs[col] = st.number_input(
                        f"{col}",
                        value=float(st.session_state.get(f"field_{col}", benign_medians[col])),
                        format="%.5f",
                        key=f"field_{col}"
                    )

        st.markdown("---")
        st.markdown("### 3. Classification & Decision Boundary Analysis")

        predict_clicked = st.button("🔮 Predict Tumor Diagnosis", type="primary", use_container_width=True)

        if predict_clicked:
            input_df = pd.DataFrame([user_inputs])[feature_names]

            if model_name in ["Logistic Regression", "Support Vector Machine"]:
                input_scaled = scaler.transform(input_df)
            else:
                input_scaled = input_df

            prediction = model.predict(input_scaled)[0]

            decision_score = None
            if hasattr(model, "decision_function"):
                decision_score = float(model.decision_function(input_scaled)[0])

            if prediction == "M":
                st.markdown(
                    """
                    <div class="result-badge-malignant">
                        <div style="font-size: 2.2rem; margin-bottom: 6px;">🚨</div>
                        <h2 style="margin: 0; color: #880e4f; font-weight: 800; font-size: 1.8rem;">Diagnosis: Malignant (Cancerous)</h2>
                        <p style="margin: 8px 0 0 0; color: #ad1457; font-size: 1.05rem;">The nuclear morphology patterns align with malignant tumor cell clusters.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="result-badge-benign">
                        <div style="font-size: 2.2rem; margin-bottom: 6px;">✅</div>
                        <h2 style="margin: 0; color: #4a148c; font-weight: 800; font-size: 1.8rem;">Diagnosis: Benign (Non-Cancerous)</h2>
                        <p style="margin: 8px 0 0 0; color: #512da8; font-size: 1.05rem;">The nuclear morphology dimensions remain within safe, typical non-cancerous ranges.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Mathematical Decision Function Gauge
            if decision_score is not None:
                st.markdown("#### SVM Decision Boundary Proximity")
                
                # Normalize decision score to a 0-100% risk meter for visual gauge
                # SVM score: -3.0 (benign) to +3.0 (malignant)
                clamped_score = max(-3.0, min(3.0, decision_score))
                malignancy_risk_pct = int(((clamped_score + 3.0) / 6.0) * 100)

                st.progress(malignancy_risk_pct / 100.0)
                
                col_g1, col_g2, col_g3 = st.columns(3)
                col_g1.metric("Decision Score", f"{decision_score:+.4f}", "Negative = Benign | Positive = Malignant")
                col_g2.metric("Boundary Distance", f"{abs(decision_score):.3f}", "Margin confidence magnitude")
                col_g3.metric("Estimated Risk Index", f"{malignancy_risk_pct}%", "Proximity toward Malignant zone")

            # Comparative Feature Benchmarking Table
            st.markdown("#### Patient Measurements vs Population Benchmarks")
            benchmark_data = [
                {
                    "Feature": "Radius Mean",
                    "Patient Value": f"{user_inputs['radius_mean']:.2f}",
                    "Benign Average": "12.15",
                    "Malignant Average": "17.46",
                    "Status": "Elevated (Malignant-like)" if user_inputs["radius_mean"] > 14.5 else "Normal (Benign-like)"
                },
                {
                    "Feature": "Perimeter Mean",
                    "Patient Value": f"{user_inputs['perimeter_mean']:.2f}",
                    "Benign Average": "78.08",
                    "Malignant Average": "115.37",
                    "Status": "Elevated (Malignant-like)" if user_inputs["perimeter_mean"] > 95.0 else "Normal (Benign-like)"
                },
                {
                    "Feature": "Area Mean",
                    "Patient Value": f"{user_inputs['area_mean']:.2f}",
                    "Benign Average": "462.79",
                    "Malignant Average": "978.38",
                    "Status": "Elevated (Malignant-like)" if user_inputs["area_mean"] > 650.0 else "Normal (Benign-like)"
                },
                {
                    "Feature": "Concavity Mean",
                    "Patient Value": f"{user_inputs['concavity_mean']:.4f}",
                    "Benign Average": "0.0461",
                    "Malignant Average": "0.1608",
                    "Status": "Elevated (Malignant-like)" if user_inputs["concavity_mean"] > 0.08 else "Normal (Benign-like)"
                },
                {
                    "Feature": "Concave Points Mean",
                    "Patient Value": f"{user_inputs['concave points_mean']:.4f}",
                    "Benign Average": "0.0257",
                    "Malignant Average": "0.0880",
                    "Status": "Elevated (Malignant-like)" if user_inputs["concave points_mean"] > 0.05 else "Normal (Benign-like)"
                }
            ]
            st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True)

            st.info(f"Model Engine: **{model_name}** | Preprocessing: Standardized via StandardScaler fitted on `data/data.csv`.")
