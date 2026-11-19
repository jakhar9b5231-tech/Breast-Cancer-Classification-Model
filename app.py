# ==============================================================================
# Project: Breast Cancer Classification Using Machine Learning
# File: app.py
# Description: Interactive Streamlit web application designed with an aesthetic
#              Cancer Awareness Foundation theme (Rose Pink, White, & Lavender).
# ==============================================================================

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration with Pink Ribbon awareness icon
st.set_page_config(
    page_title="Breast Cancer Classification | Awareness Dashboard",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# CANCER FOUNDATION THEME: CUSTOM CSS INJECTION
# Palette: Rose Pink (#E91E63, #F06292), Soft Pink Mist (#FDF2F5),
#          Lavender (#7E57C2, #EDE7F6, #D1C4E9), Crisp White (#FFFFFF)
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Background and Typography */
    .stApp {
        background-color: #FAF7FB;
        color: #2D3748;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FDF4F8 0%, #F5EEF9 100%) !important;
        border-right: 1px solid #ECD9EC;
    }

    /* Awareness Ribbon Header Banner */
    .ribbon-banner {
        background: linear-gradient(135deg, #E91E63 0%, #9C27B0 100%);
        color: white;
        padding: 24px 28px;
        border-radius: 14px;
        box-shadow: 0 6px 20px rgba(233, 30, 99, 0.22);
        margin-bottom: 24px;
    }
    .ribbon-banner h1 {
        color: #FFFFFF !important;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .ribbon-banner p {
        color: #FCE4EC !important;
        font-size: 1.02rem;
        margin: 0;
        line-height: 1.5;
    }

    /* Card Styling */
    .pink-card {
        background: #FFFFFF;
        border: 1px solid #F3D8E5;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 4px 14px rgba(233, 30, 99, 0.06);
        margin-bottom: 18px;
    }
    .lavender-card {
        background: #FFFFFF;
        border: 1px solid #E1D4EC;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 4px 14px rgba(126, 87, 194, 0.06);
        margin-bottom: 18px;
    }

    /* Metric Card Customization */
    .metric-badge {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF5F8 100%);
        border: 1px solid #F8BBD0;
        border-left: 5px solid #E91E63;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 3px 10px rgba(233, 30, 99, 0.05);
    }
    .metric-badge-lavender {
        background: linear-gradient(135deg, #FFFFFF 0%, #F6F2FB 100%);
        border: 1px solid #D1C4E9;
        border-left: 5px solid #7E57C2;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 3px 10px rgba(126, 87, 194, 0.05);
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #718096;
        margin-bottom: 4px;
    }
    .metric-number {
        font-size: 1.9rem;
        font-weight: 700;
        color: #C2185B;
    }
    .metric-number-lavender {
        font-size: 1.9rem;
        font-weight: 700;
        color: #5E35B1;
    }

    /* Callout & Alert Boxes */
    .malignant-alert {
        background-color: #FFF0F4;
        border: 2px solid #F8BBD0;
        border-left: 6px solid #E91E63;
        border-radius: 10px;
        padding: 18px 22px;
        color: #880E4F;
        margin-top: 15px;
    }
    .benign-alert {
        background-color: #F8F4FD;
        border: 2px solid #D1C4E9;
        border-left: 6px solid #7E57C2;
        border-radius: 10px;
        padding: 18px 22px;
        color: #311B92;
        margin-top: 15px;
    }
    .disclaimer-badge {
        background: #FFF9E6;
        border-left: 4px solid #FFA000;
        border-radius: 8px;
        padding: 12px 18px;
        font-size: 0.88rem;
        color: #795548;
        line-height: 1.5;
        margin-top: 15px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 2px solid #EAD8EB;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 18px;
        font-weight: 600;
        color: #6A1B9A;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #E91E63 !important;
        border-bottom: 3px solid #E91E63 !important;
    }

    /* Primary Predict Button */
    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="baseButton-primary"],
    button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #E91E63 0%, #8E24AA 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 10px 32px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(233, 30, 99, 0.3) !important;
        transition: all 0.25s ease !important;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="baseButton-primary"]:hover,
    button[data-testid="baseButton-primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(233, 30, 99, 0.45) !important;
    }

    /* Secondary Sample Loader Buttons */
    div.stButton > button[kind="secondary"],
    div.stButton > button[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-secondary"] {
        background-color: #FFFFFF !important;
        color: #5E35B1 !important;
        border: 1.5px solid #D1C4E9 !important;
        border-radius: 20px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="secondary"]:hover,
    div.stButton > button[data-testid="baseButton-secondary"]:hover,
    button[data-testid="baseButton-secondary"]:hover {
        background-color: #F3E5F5 !important;
        border-color: #BA68C8 !important;
        color: #4A148C !important;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# DATA & MODEL LOADING FUNCTIONS (CACHED)
# ------------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    """Loads dataset from data/data.csv with fallback paths."""
    candidates = [
        os.path.join("data", "data.csv"),
        os.path.join("..", "data", "data.csv"),
        "data.csv",
        os.path.join("programs", "data.csv")
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


@st.cache_resource
def load_model_artifacts():
    """Loads trained model artifacts saved by train_model.py."""
    candidates = [
        "model.joblib",
        os.path.join("..", "model.joblib"),
        os.path.join("programs", "model.joblib")
    ]
    for path in candidates:
        if os.path.exists(path):
            return joblib.load(path)
    return None


df_raw = load_dataset()
artifacts = load_model_artifacts()


# ------------------------------------------------------------------------------
# SIDEBAR NAVIGATION & THEME BRANDING
# ------------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0 16px 0;">
    <div style="font-size: 2.4rem; margin-bottom: 4px;">🎗️</div>
    <div style="font-weight: 800; font-size: 1.18rem; color: #AD1457; letter-spacing: 0.3px;">
        Hope & Science AI
    </div>
    <div style="font-size: 0.82rem; color: #6A1B9A; font-weight: 500;">
        Breast Cancer ML Classification
    </div>
</div>
""", unsafe_allow_html=True)

menu_choice = st.sidebar.radio(
    "Explore Dashboard:",
    [
        "🎗️ Project Overview",
        "📊 Interactive Dataset Explorer",
        "🔬 Interactive Exploratory Data Analysis",
        "⚖️ Model Benchmark & Evaluation",
        "🔮 Real-Time Cancer Prediction"
    ]
)

st.sidebar.markdown("---")

# Quick Stats Ribbon on Sidebar
if df_raw is not None:
    benign_count = int((df_raw["diagnosis"] == "B").sum())
    malignant_count = int((df_raw["diagnosis"] == "M").sum())
    st.sidebar.markdown(f"""
    <div style="background: white; border-radius: 10px; border: 1px solid #F3D8E5; padding: 14px; margin-bottom: 12px;">
        <div style="font-weight: 700; color: #880E4F; font-size: 0.88rem; margin-bottom: 6px;">
            Dataset Snapshot
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.84rem; color: #4A148C; margin-bottom: 4px;">
            <span>🌸 Benign Cases:</span> <b>{benign_count} ({benign_count/len(df_raw)*100:.1f}%)</b>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.84rem; color: #C2185B;">
            <span>🌹 Malignant Cases:</span> <b>{malignant_count} ({malignant_count/len(df_raw)*100:.1f}%)</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.info(
    "**Course Project:** College ML Viva Demonstration\n\n"
    "**Core Algorithms:**\n"
    "- Gaussian Naive Bayes\n"
    "- Logistic Regression\n"
    "- Support Vector Machine (SVM)"
)


# ==============================================================================
# SECTION 1: PROJECT OVERVIEW
# ==============================================================================
if menu_choice == "🎗️ Project Overview":
    st.markdown("""
    <div class="ribbon-banner">
        <h1>🎗️ Breast Cancer Classification Using Machine Learning</h1>
        <p>A machine learning diagnostic study inspired by breast cancer research foundations —
        applying transparent, explainable AI algorithms to differentiate benign breast masses from malignant tumors.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.8, 1.2])

    with col1:
        st.markdown("""
        <div class="pink-card">
            <h3 style="color: #AD1457; margin-top: 0;">🌸 Clinical Context & Project Purpose</h3>
            <p style="line-height: 1.6; color: #4A5568;">
                Breast cancer is the most frequently diagnosed life-threatening cancer in women worldwide.
                Accurate, early diagnosis through <b>Fine Needle Aspirate (FNA)</b> biopsies drastically improves treatment success
                and patient survival rates.
            </p>
            <p style="line-height: 1.6; color: #4A5568;">
                This project implements, evaluates, and compares three core machine learning classifiers taught in introductory AI:
            </p>
            <ul style="color: #4A5568; line-height: 1.7;">
                <li><b>Gaussian Naive Bayes:</b> Probabilistic baseline modeling Gaussian likelihood functions.</li>
                <li><b>Logistic Regression:</b> Linear margin decision boundaries with sigmoid probabilities.</li>
                <li><b>Support Vector Machine (SVM):</b> Maximum-margin hyperplanes in normalized feature space.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="lavender-card">
            <h3 style="color: #5E35B1; margin-top: 0;">🔬 Project Objectives</h3>
            <ul style="color: #4A5568; line-height: 1.8;">
                <li><b>Data Hygiene:</b> Safely clean identifier artifacts (<code>id</code>, <code>Unnamed: 32</code>) and prevent data leakage via training-only <code>StandardScaler</code> fits.</li>
                <li><b>Clinical Prioritization:</b> Optimize for <b>Recall on Malignant tumors</b> because missed cancer cases (False Negatives) carry severe clinical risk.</li>
                <li><b>Viva Transparency:</b> Maintain transparent mathematical metrics and real-time decision boundary score visualization.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #FFF0F5 0%, #EDE7F6 100%); border: 1px solid #F3D8E5; border-radius: 12px; padding: 22px; margin-bottom: 18px;">
            <h4 style="color: #880E4F; margin-top: 0; font-weight: 700;">💖 Cancer Awareness Palette</h4>
            <div style="display: flex; gap: 8px; margin-bottom: 12px;">
                <div style="flex: 1; height: 35px; background: #E91E63; border-radius: 6px;" title="Ribbon Pink"></div>
                <div style="flex: 1; height: 35px; background: #F06292; border-radius: 6px;" title="Pastel Rose"></div>
                <div style="flex: 1; height: 35px; background: #7E57C2; border-radius: 6px;" title="Awareness Lavender"></div>
                <div style="flex: 1; height: 35px; background: #D1C4E9; border-radius: 6px;" title="Lilac Mist"></div>
            </div>
            <p style="font-size: 0.88rem; color: #555; line-height: 1.5;">
                Styled in honor of cancer foundation colors — symbolizing <b>compassion, strength, and scientific hope</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer-badge">
            <b>⚠️ Academic Disclaimer:</b><br>
            This application is created exclusively as an undergraduate engineering/college coursework project.
            It is designed for study and presentation purposes and is <b>not a clinical diagnostic device</b>.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 2: INTERACTIVE DATASET EXPLORER
# ==============================================================================
elif menu_choice == "📊 Interactive Dataset Explorer":
    st.markdown("""
    <div class="ribbon-banner">
        <h1>📊 Interactive Dataset Explorer</h1>
        <p>Explore the Wisconsin Diagnostic Breast Cancer (WDBC) dataset loaded directly from <code>data/data.csv</code>.</p>
    </div>
    """, unsafe_allow_html=True)

    if df_raw is None:
        st.error("Dataset `data/data.csv` could not be located.")
    else:
        # Top KPI Cards with Pink/Lavender Ribbon Styling
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        total_rows = len(df_raw)
        b_count = int((df_raw["diagnosis"] == "B").sum())
        m_count = int((df_raw["diagnosis"] == "M").sum())

        with kpi1:
            st.markdown(f"""
            <div class="metric-badge">
                <div class="metric-title">Total Biopsy Samples</div>
                <div class="metric-number">{total_rows}</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
            <div class="metric-badge">
                <div class="metric-title">Benign Cases (B)</div>
                <div class="metric-number">{b_count} <span style="font-size: 0.95rem; font-weight: 500;">({b_count/total_rows*100:.1f}%)</span></div>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
            <div class="metric-badge">
                <div class="metric-title">Malignant Cases (M)</div>
                <div class="metric-number" style="color: #D81B60;">{m_count} <span style="font-size: 0.95rem; font-weight: 500;">({m_count/total_rows*100:.1f}%)</span></div>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""
            <div class="metric-badge-lavender">
                <div class="metric-title">Predictive Features</div>
                <div class="metric-number-lavender">30 <span style="font-size: 0.95rem; font-weight: 500;">Nuclear Attributes</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Interactive Filtering Controls
        st.markdown("""<div class="pink-card">""", unsafe_allow_html=True)
        st.subheader("🔍 Filter & Inspect Dataset Records")
        filter_col1, filter_col2, filter_col3 = st.columns([1.2, 1.5, 1])

        with filter_col1:
            diag_filter = st.radio(
                "Filter by Diagnosis:",
                ["All Cases (569)", "Benign Only (357)", "Malignant Only (212)"],
                horizontal=True
            )

        with filter_col2:
            feat_category = st.selectbox(
                "Feature Subset to Display:",
                ["Mean Features (10)", "Standard Error Features (10)", "Worst Features (10)", "All 30 Features"]
            )

        with filter_col3:
            rows_to_show = st.slider("Rows to Preview:", min_value=5, max_value=50, value=10, step=5)

        # Apply filtering
        filtered_df = df_raw.copy()
        if "Benign" in diag_filter:
            filtered_df = filtered_df[filtered_df["diagnosis"] == "B"]
        elif "Malignant" in diag_filter:
            filtered_df = filtered_df[filtered_df["diagnosis"] == "M"]

        base_cols = ["id", "diagnosis"] if "id" in filtered_df.columns else ["diagnosis"]
        if feat_category == "Mean Features (10)":
            selected_cols = base_cols + [c for c in filtered_df.columns if "_mean" in c]
        elif feat_category == "Standard Error Features (10)":
            selected_cols = base_cols + [c for c in filtered_df.columns if "_se" in c]
        elif feat_category == "Worst Features (10)":
            selected_cols = base_cols + [c for c in filtered_df.columns if "_worst" in c]
        else:
            selected_cols = [c for c in filtered_df.columns if c != "Unnamed: 32"]

        st.dataframe(filtered_df[selected_cols].head(rows_to_show), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Statistical Summary with Toggle
        with st.expander("📈 View 8-Point Statistical Summary (`df.describe()`)", expanded=False):
            st.dataframe(df_raw.describe(), use_container_width=True)

        # Data Cleaning Summary Card
        st.markdown("""
        <div class="lavender-card">
            <h4 style="color: #4A148C; margin-top: 0;">🧹 Data Cleaning Summary for Viva</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; font-size: 0.92rem; color: #4A5568;">
                <div>• <b>id Column:</b> Dropped because patient IDs are arbitrary serial numbers with zero predictive pathology value.</div>
                <div>• <b>Unnamed: 32:</b> Dropped because it contains 100% NaN values (caused by a trailing comma in the Kaggle CSV).</div>
                <div>• <b>Duplicate Check:</b> Verified exactly 0 duplicate rows across all 569 patient records.</div>
                <div>• <b>Missing Values:</b> 0 missing values across all 30 analytical cell measurements.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 3: INTERACTIVE EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
elif menu_choice == "🔬 Interactive Exploratory Data Analysis":
    st.markdown("""
    <div class="ribbon-banner">
        <h1>🔬 Interactive Exploratory Data Analysis (EDA)</h1>
        <p>Visually discover patterns, correlations, and cluster separations across nuclear cell dimensions.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_preset, tab_live = st.tabs(["🌸 Curated EDA Visualizations", "✨ Dynamic Scatter & Correlation Explorer"])

    with tab_preset:
        eda_choice = st.selectbox(
            "Select an Analysis View to Inspect:",
            [
                "1. Target Class Distribution (Benign vs Malignant)",
                "2. Feature Correlation Heatmap (Multicollinearity Analysis)",
                "3. Nuclear Dimension Histograms (KDE Curves)",
                "4. Feature Box Plots Grouped by Diagnosis",
                "5. 2D Scatter Plot: Radius vs Texture"
            ]
        )

        img_map = {
            "1. Target Class Distribution (Benign vs Malignant)": ("plots/target_distribution.png",
                "<b>Observation:</b> The dataset contains 357 Benign (62.7%) and 212 Malignant (37.3%) tumors. "
                "Because of this class imbalance, we utilize <code>stratify=y</code> during train-test splitting to maintain identical class ratios in both partitions."),
            "2. Feature Correlation Heatmap (Multicollinearity Analysis)": ("plots/correlation_heatmap.png",
                "<b>Observation:</b> Geometric attributes such as <code>radius_mean</code>, <code>perimeter_mean</code>, and <code>area_mean</code> "
                "exhibit near-perfect correlation (r ≈ 0.99). This multicollinearity violates the conditional independence assumption of Naive Bayes, "
                "which is why Naive Bayes records a slightly lower recall compared to SVM."),
            "3. Nuclear Dimension Histograms (KDE Curves)": ("plots/feature_histograms.png",
                "<b>Observation:</b> Malignant cell nuclei (red) display significantly elevated radii, perimeters, and areas compared to benign cells (green). "
                "This clear bimodal distribution provides strong discriminative signal for linear and margin classifiers."),
            "4. Feature Box Plots Grouped by Diagnosis": ("plots/feature_boxplots.png",
                "<b>Observation:</b> The median and interquartile range (IQR) for malignant cases are noticeably higher with wider variance, "
                "confirming that malignant cells exhibit pleomorphism (variation in size and shape)."),
            "5. 2D Scatter Plot: Radius vs Texture": ("plots/scatter_plot.png",
                "<b>Observation:</b> Projecting tumors onto the <code>radius_mean</code> vs <code>texture_mean</code> plane reveals clear cluster separation, "
                "explaining why linear decision boundaries and SVM hyperplanes achieve >96% accuracy.")
        }

        img_file, obs_text = img_map[eda_choice]
        if os.path.exists(img_file):
            col_img, col_info = st.columns([1.6, 1])
            with col_img:
                st.image(img_file, use_container_width=True)
            with col_info:
                st.markdown(f"""
                <div class="pink-card" style="margin-top: 15px;">
                    <h4 style="color: #AD1457; margin-top: 0;">💡 Clinical & Viva Insight</h4>
                    <p style="line-height: 1.6; color: #4A5568;">{obs_text}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning(f"Visualization image `{img_file}` not found. Run `python train_model.py` to generate plots.")

    with tab_live:
        st.markdown("""
        <div class="lavender-card">
            <h4 style="color: #5E35B1; margin-top: 0;">✨ Build Your Own Live Scatter Plot</h4>
            <p style="color: #666; font-size: 0.9rem;">
                Select any two physical cell features from <code>data.csv</code> to inspect their interactive decision boundary separation in real time.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if df_raw is not None:
            clean_df = df_raw.drop(columns=[c for c in ["id", "Unnamed: 32"] if c in df_raw.columns])
            num_cols = [c for c in clean_df.columns if c != "diagnosis"]

            s_col1, s_col2, s_col3 = st.columns([1, 1, 1])
            with s_col1:
                x_axis = st.selectbox("X-Axis Feature:", num_cols, index=num_cols.index("radius_mean") if "radius_mean" in num_cols else 0)
            with s_col2:
                y_axis = st.selectbox("Y-Axis Feature:", num_cols, index=num_cols.index("texture_mean") if "texture_mean" in num_cols else 1)
            with s_col3:
                show_kde = st.checkbox("Show Density Contours", value=False)

            # Live Seaborn/Matplotlib Plot Styled in Foundation Pink & Lavender
            fig, ax = plt.subplots(figsize=(8, 4.8))
            palette_awareness = {"B": "#7E57C2", "M": "#E91E63"}

            if show_kde:
                sns.kdeplot(data=clean_df, x=x_axis, y=y_axis, hue="diagnosis", palette=palette_awareness, levels=5, alpha=0.5, ax=ax)
            sns.scatterplot(
                data=clean_df, x=x_axis, y=y_axis, hue="diagnosis",
                palette=palette_awareness, alpha=0.85, s=45, ax=ax
            )

            ax.set_title(f"Live Clinical Scatter: {x_axis} vs {y_axis}", fontsize=12, fontweight="bold", color="#880E4F")
            ax.set_xlabel(x_axis, fontweight="600")
            ax.set_ylabel(y_axis, fontweight="600")
            ax.grid(True, linestyle="--", alpha=0.4)
            handles, labels = ax.get_legend_handles_labels()
            labels = ["Benign (Non-Cancerous)" if l == "B" else "Malignant (Cancerous)" for l in labels]
            ax.legend(handles=handles, labels=labels, title="Diagnosis")
            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)


# ==============================================================================
# SECTION 4: MODEL BENCHMARK & EVALUATION
# ==============================================================================
elif menu_choice == "⚖️ Model Benchmark & Evaluation":
    st.markdown("""
    <div class="ribbon-banner">
        <h1>⚖️ Model Benchmark & Evaluation</h1>
        <p>Rigorous comparative assessment across Gaussian Naive Bayes, Logistic Regression, and Support Vector Machine.</p>
    </div>
    """, unsafe_allow_html=True)

    if artifacts is None:
        st.error("Model artifacts not found. Please run `python train_model.py` first.")
    else:
        comparison_df = artifacts["comparison_table"]

        # 3 Benchmark Cards
        c1, c2, c3 = st.columns(3)
        with c1:
            gnb_acc = comparison_df.loc[comparison_df["Model"] == "Gaussian Naive Bayes", "Accuracy"].values[0]
            st.markdown(f"""
            <div class="lavender-card" style="border-top: 4px solid #7E57C2;">
                <div class="metric-title">Gaussian Naive Bayes</div>
                <div class="metric-number-lavender">{gnb_acc*100:.2f}%</div>
                <div style="font-size: 0.82rem; color: #666; margin-top: 4px;">Baseline • Unscaled Features</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            lr_acc = comparison_df.loc[comparison_df["Model"] == "Logistic Regression", "Accuracy"].values[0]
            st.markdown(f"""
            <div class="pink-card" style="border-top: 4px solid #F06292;">
                <div class="metric-title">Logistic Regression</div>
                <div class="metric-number" style="color: #D81B60;">{lr_acc*100:.2f}%</div>
                <div style="font-size: 0.82rem; color: #666; margin-top: 4px;">Linear Boundary • StandardScaled</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            svm_acc = comparison_df.loc[comparison_df["Model"] == "Support Vector Machine", "Accuracy"].values[0]
            st.markdown(f"""
            <div class="pink-card" style="border-top: 4px solid #E91E63; background: #FFF5F8;">
                <div class="metric-title">Support Vector Machine (Top)</div>
                <div class="metric-number" style="color: #C2185B;">{svm_acc*100:.2f}%</div>
                <div style="font-size: 0.82rem; color: #C2185B; font-weight: 600; margin-top: 4px;">Selected Model • F1: 0.9630</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Performance DataFrame Table
        st.subheader("📋 Comprehensive Evaluation Metrics (Positive Class = 'M')")
        formatted_table = comparison_df.copy()
        st.dataframe(formatted_table.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}"
        }).highlight_max(subset=["Accuracy", "Recall", "F1-Score"], color="#FCE4EC"), use_container_width=True)

        # Side-by-side Plots
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            st.subheader("📊 Accuracy Comparison Bar Chart")
            if os.path.exists("plots/model_comparison.png"):
                st.image("plots/model_comparison.png", use_container_width=True)
        with col_plot2:
            st.subheader("🎯 Confusion Matrices Comparison")
            if os.path.exists("plots/confusion_matrices.png"):
                st.image("plots/confusion_matrices.png", use_container_width=True)

        # Critical Viva Explanations
        st.markdown("""
        <div class="lavender-card">
            <h4 style="color: #4A148C; margin-top: 0;">🩺 Clinical Significance: Why False Negatives Matter</h4>
            <div style="color: #4A5568; line-height: 1.6;">
                In oncology screening, classification errors carry drastically asymmetric consequences:
                <ul>
                    <li><b>False Positive (FP):</b> A benign cyst is predicted as malignant. The patient experiences temporary anxiety and is referred for secondary screening (ultrasound/biopsy) which resolves the error.</li>
                    <li><b>False Negative (FN):</b> A cancerous malignant tumor is mistakenly classified as benign. The patient is falsely reassured, treatment is delayed, and the cancer progresses undetected.</li>
                </ul>
                Therefore, <b>high Recall (Sensitivity) for class 'M'</b> is the most critical metric in breast cancer detection.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 5: REAL-TIME CANCER PREDICTION
# ==============================================================================
elif menu_choice == "🔮 Real-Time Cancer Prediction":
    st.markdown("""
    <div class="ribbon-banner">
        <h1>🔮 Real-Time Cancer Prediction Interface</h1>
        <p>Interactive diagnostic console — test live patient measurements, compare multi-model predictions, and inspect boundary confidence.</p>
    </div>
    """, unsafe_allow_html=True)

    if artifacts is None:
        st.error("Model artifacts not found. Please train models using `python train_model.py` first.")
    else:
        feature_names = artifacts["feature_names"]
        scaler = artifacts["scaler"]
        primary_model = artifacts["model"]
        primary_model_name = artifacts["model_name"]
        all_models = artifacts.get("all_models", {primary_model_name: primary_model})

        # ----------------------------------------------------------------------
        # REFERENCE CLINICAL PRESET VALUES
        # ----------------------------------------------------------------------
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

        def set_preset_features(sample_dict, label):
            st.session_state["active_sample_label"] = label
            for col in feature_names:
                st.session_state[f"f_{col}"] = float(sample_dict.get(col, benign_medians.get(col, 0.0)))
            st.session_state["has_run_prediction"] = True

        # Initialize session state for all 30 features
        for col in feature_names:
            if f"f_{col}" not in st.session_state:
                st.session_state[f"f_{col}"] = float(benign_medians[col])

        # ----------------------------------------------------------------------
        # ONE-CLICK SAMPLE LOADERS (PINK & LAVENDER THEMED BUTTONS)
        # ----------------------------------------------------------------------
        st.markdown("""<div class="pink-card">""", unsafe_allow_html=True)
        st.subheader("1. Quick Patient Case Loaders (Viva Demonstration)")
        st.write("Click any case to populate all 30 analytical features directly from confirmed dataset entries:")

        b_c1, b_c2, b_c3, b_c4 = st.columns(4)
        with b_c1:
            if st.button("🌹 Load Malignant (Row 0)", type="secondary", use_container_width=True):
                set_preset_features(sample_malignant_row0, "Row 0: Confirmed Malignant")
                st.rerun()
        with b_c2:
            if st.button("🌸 Load Benign (Row 19)", type="secondary", use_container_width=True):
                set_preset_features(sample_benign_row19, "Row 19: Confirmed Benign")
                st.rerun()
        with b_c3:
            if st.button("🌸 Load Benign (Row 20)", type="secondary", use_container_width=True):
                set_preset_features(sample_benign_row20, "Row 20: Confirmed Benign")
                st.rerun()
        with b_c4:
            if st.button("🔄 Reset (Benign Medians)", type="secondary", use_container_width=True):
                set_preset_features(benign_medians, "Benign Population Baseline (Medians)")
                st.rerun()

        active_lbl = st.session_state.get("active_sample_label", "Benign Population Baseline (Medians)")
        if "Malignant" in active_lbl:
            st.markdown(f"**Loaded Case:** <span style='color: #C2185B; font-weight: 700;'>{active_lbl}</span> *(Expected: Malignant / Cancerous)*", unsafe_allow_html=True)
        else:
            st.markdown(f"**Loaded Case:** <span style='color: #5E35B1; font-weight: 700;'>{active_lbl}</span> *(Expected: Benign / Non-Cancerous)*", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # MULTI-MODEL COMPARISON SELECTOR
        # ----------------------------------------------------------------------
        st.markdown("""<div class="lavender-card">""", unsafe_allow_html=True)
        st.subheader("2. Select Machine Learning Model for Prediction")
        model_options = list(all_models.keys()) + ["🤝 Multi-Model Consensus (All 3 Models)"]
        selected_model_option = st.selectbox(
            "Choose Algorithm to Run:",
            model_options,
            index=model_options.index(primary_model_name) if primary_model_name in model_options else 0
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # 30-FEATURE INPUT CONSOLE WITH ACCORDION TABS
        # ----------------------------------------------------------------------
        st.subheader("3. Analytical Nuclear Feature Measurements")

        with st.expander("📖 Clinical Feature Guide (Click to learn what features mean for viva)", expanded=False):
            st.markdown("""
            - **Radius / Perimeter / Area:** Fundamental physical size of the cell nucleus. Malignant cells typically show enlarged dimensions.
            - **Texture:** Variation in gray-scale intensities on FNA microscopy.
            - **Smoothness:** Local variation in radius lengths along the nuclear perimeter.
            - **Compactness:** Defined as $(perimeter^2 / area) - 1.0$, measures irregularity of nuclear boundary.
            - **Concavity & Concave Points:** Severity and number of indentations in the contour of the cell nucleus.
            - **Symmetry & Fractal Dimension:** Nuclear symmetry and coastline approximation ('fractal boundary').
            """)

        tab_m, tab_s, tab_w = st.tabs(["Mean Dimensions (10)", "Standard Error (10)", "Worst Dimensions (10)"])

        user_input_dict = {}

        with tab_m:
            st.caption("Average cell nucleus measurements:")
            m_cols = [c for c in feature_names if "_mean" in c]
            c_m1, c_m2 = st.columns(2)
            for i, col in enumerate(m_cols):
                with (c_m1 if i % 2 == 0 else c_m2):
                    user_input_dict[col] = st.number_input(
                        f"{col}",
                        format="%.5f",
                        key=f"f_{col}"
                    )

        with tab_s:
            st.caption("Standard error (variation) across aspirated cells:")
            s_cols = [c for c in feature_names if "_se" in c]
            c_s1, c_s2 = st.columns(2)
            for i, col in enumerate(s_cols):
                with (c_s1 if i % 2 == 0 else c_s2):
                    user_input_dict[col] = st.number_input(
                        f"{col}",
                        format="%.5f",
                        key=f"f_{col}"
                    )

        with tab_w:
            st.caption("Largest / worst measurements observed across cells:")
            w_cols = [c for c in feature_names if "_worst" in c]
            c_w1, c_w2 = st.columns(2)
            for i, col in enumerate(w_cols):
                with (c_w1 if i % 2 == 0 else c_w2):
                    user_input_dict[col] = st.number_input(
                        f"{col}",
                        format="%.5f",
                        key=f"f_{col}"
                    )

        # ----------------------------------------------------------------------
        # EXECUTE PREDICTION & INTERACTIVE DASHBOARD REPORT
        # ----------------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("🔮 Run Diagnostic Prediction", type="primary", use_container_width=True)

        if predict_clicked:
            st.session_state["has_run_prediction"] = True

        # Run diagnosis when requested or automatically on sample load
        if st.session_state.get("has_run_prediction", True):
            input_df = pd.DataFrame([user_input_dict])[feature_names]
            input_scaled = scaler.transform(input_df)

            st.markdown("---")
            st.subheader("Diagnostic Classification Results")

            # Collect predictions and probabilities for all models
            all_model_preds = {}
            all_model_probs = {}
            all_model_dscores = {}

            for m_name, clf in all_models.items():
                cur_in = input_df if m_name == "Gaussian Naive Bayes" else input_scaled
                p = clf.predict(cur_in)[0]
                all_model_preds[m_name] = p

                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(cur_in)[0]
                    m_idx = list(clf.classes_).index("M") if hasattr(clf, "classes_") else 1
                    all_model_probs[m_name] = float(probs[m_idx])
                elif hasattr(clf, "decision_function"):
                    dfunc = float(clf.decision_function(cur_in)[0])
                    all_model_dscores[m_name] = dfunc
                    all_model_probs[m_name] = float(1.0 / (1.0 + np.exp(-dfunc)))

            if "Support Vector Machine" in all_models and hasattr(all_models["Support Vector Machine"], "decision_function"):
                all_model_dscores["Support Vector Machine"] = float(all_models["Support Vector Machine"].decision_function(input_scaled)[0])

            # Determine primary prediction and display
            if selected_model_option == "🤝 Multi-Model Consensus (All 3 Models)":
                st.markdown("#### 🤝 Multi-Model Consensus Breakdown")
                mc1, mc2, mc3 = st.columns(3)
                cols_list = [mc1, mc2, mc3]

                for idx, (m_name, pred) in enumerate(all_model_preds.items()):
                    with cols_list[idx]:
                        badge_color = "#E91E63" if pred == "M" else "#7E57C2"
                        badge_text = "🚨 Malignant" if pred == "M" else "✅ Benign"
                        prob_m = all_model_probs[m_name]
                        prob_pct = (prob_m if pred == "M" else (1.0 - prob_m)) * 100.0
                        st.markdown(f"""
                        <div style="background: white; border: 2px solid {badge_color}; border-radius: 12px; padding: 16px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                            <div style="font-size: 0.85rem; font-weight: 700; color: #555;">{m_name}</div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: {badge_color}; margin: 8px 0;">{badge_text}</div>
                            <div style="font-size: 0.88rem; color: #666; font-weight: 600;">Confidence: {prob_pct:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                m_count = sum(1 for p in all_model_preds.values() if p == "M")
                b_count = sum(1 for p in all_model_preds.values() if p == "B")
                primary_pred = "M" if m_count >= 2 else "B"
                active_prob_m = float(np.mean(list(all_model_probs.values())))

                if m_count == 3 or b_count == 3:
                    st.success(f"✨ **Unanimous Consensus (3/3 Models Agree):** All models converged on **{'Malignant' if primary_pred == 'M' else 'Benign'}**.")
                else:
                    st.info(f"⚖️ **Majority Consensus (2/3 Models):** Decision determined by majority vote (**{'Malignant' if primary_pred == 'M' else 'Benign'}**).")
            else:
                primary_pred = all_model_preds[selected_model_option]
                active_prob_m = all_model_probs.get(selected_model_option, 0.5)

            # Primary Diagnosis Card
            if primary_pred == "M":
                st.markdown("""
                <div class="malignant-alert">
                    <h3 style="margin-top: 0; color: #880E4F;">🚨 Clinical Prediction: MALIGNANT (Cancerous Tumor)</h3>
                    <p style="margin-bottom: 0; line-height: 1.6;">
                        The model detected distinct morphological abnormalities in nuclear dimensions
                        (elevated cell radius, perimeter, area, and concavity) that strongly correspond to malignant breast pathology.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="benign-alert">
                    <h3 style="margin-top: 0; color: #311B92;">✅ Clinical Prediction: BENIGN (Non-Cancerous Mass)</h3>
                    <p style="margin-bottom: 0; line-height: 1.6;">
                        The morphological properties of the cell nuclei fall safely within normal benign parameters.
                        Physical measurements do not exhibit the irregular enlargement typical of malignant tissue.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Mathematical Boundary Distance & Confidence Analysis
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📐 Mathematical Decision Boundary & Confidence Analysis")

            g_col1, g_col2 = st.columns([1.5, 1])

            with g_col1:
                st.progress(float(active_prob_m))
                st.caption(f"Estimated Malignant Probability: **{active_prob_m*100:.1f}%** | Benign Probability: **{(1.0 - active_prob_m)*100:.1f}%**")

                if selected_model_option == "Gaussian Naive Bayes":
                    st.write("**Posterior Probability Formula:** `P(Diagnosis = M | X)`")
                    st.caption("Gaussian Naive Bayes computes posterior likelihood using Bayes' rule with continuous Gaussian density functions.")
                elif selected_model_option == "Logistic Regression":
                    d_score = all_model_dscores.get("Logistic Regression", 0.0)
                    st.write(f"**Logistic Regression Log-Odds (z):** `{d_score:+.4f}`")
                    st.caption("Mapped through standard logistic sigmoid: `p = 1 / (1 + e^-z)`.")
                elif selected_model_option == "Support Vector Machine":
                    d_score = all_model_dscores.get("Support Vector Machine", 0.0)
                    st.write(f"**SVM Hyperplane Margin Distance Score:** `{d_score:+.4f}`")
                    st.caption("Negative score (< 0) indicates Benign margin space; Positive score (> 0) indicates Malignant margin space.")
                else:
                    st.write("**Ensemble Consensus:** Average Malignant Risk Across All 3 Classifiers")
                    st.caption("Combines Gaussian Naive Bayes, Logistic Regression, and Support Vector Machine.")

            with g_col2:
                if selected_model_option == "Gaussian Naive Bayes":
                    viva_text = "<b>Viva Note:</b> Gaussian Naive Bayes assumes independent features conditional on class: <code>P(X|y) = ∏ P(x_i|y)</code>, where each continuous feature follows a normal distribution."
                elif selected_model_option == "Logistic Regression":
                    viva_text = "<b>Viva Note:</b> Logistic Regression defines a linear boundary <code>z = w · x + b</code>. The sign and magnitude of <code>z</code> reflect log-odds of malignancy."
                elif selected_model_option == "Support Vector Machine":
                    viva_text = "<b>Viva Note:</b> Support Vector Machine computes the signed geometric distance from the separating hyperplane: <code>f(x) = w · x + b</code> with maximum margin."
                else:
                    viva_text = "<b>Viva Note:</b> Multi-model consensus aggregates generative (Naive Bayes), discriminative probabilistic (Logistic Regression), and margin-based (SVM) paradigms."

                st.markdown(f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px; font-size: 0.85rem; color: #555; line-height: 1.5;">
                    {viva_text}
                </div>
                """, unsafe_allow_html=True)

            # Interactive Patient Comparison vs Population Averages
            st.markdown("#### 🔍 Patient Measurement Comparison vs Clinical Dataset")
            comp_features = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "concavity_mean", "concave points_mean"]
            if df_raw is not None:
                b_means = df_raw[df_raw["diagnosis"] == "B"][comp_features].mean()
                m_means = df_raw[df_raw["diagnosis"] == "M"][comp_features].mean()

                comp_data = []
                for f in comp_features:
                    patient_val = float(user_input_dict[f])
                    b_val = float(b_means[f])
                    m_val = float(m_means[f])
                    dist_to_b = abs(patient_val - b_val)
                    dist_to_m = abs(patient_val - m_val)
                    closest = "Benign Range 🌸" if dist_to_b < dist_to_m else "Malignant Range 🌹"
                    status = "Above Benign Avg" if patient_val > b_val else "Within Benign Avg"

                    comp_data.append({
                        "Feature": f,
                        "Patient Value": f"{patient_val:.4f}",
                        "Benign Pop Mean": f"{b_val:.4f}",
                        "Malignant Pop Mean": f"{m_val:.4f}",
                        "Relative to Benign": status,
                        "Closest Profile": closest
                    })

                st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

                # Interactive Relative Comparison Chart
                st.markdown("##### 📊 Relative Feature Levels Normalized to Benign Baseline")
                fig_comp, ax_comp = plt.subplots(figsize=(9, 3.8))
                x_indices = np.arange(len(comp_features))
                bar_width = 0.25

                p_vals = [float(user_input_dict[f]) for f in comp_features]
                b_vals = [float(b_means[f]) for f in comp_features]
                m_vals = [float(m_means[f]) for f in comp_features]

                p_norm = [p / (b if b != 0 else 1.0) for p, b in zip(p_vals, b_vals)]
                b_norm = [1.0] * len(comp_features)
                m_norm = [m / (b if b != 0 else 1.0) for m, b in zip(m_vals, b_vals)]

                ax_comp.bar(x_indices - bar_width, b_norm, bar_width, label="Benign Baseline (1.0x)", color="#7E57C2", alpha=0.85)
                ax_comp.bar(x_indices, p_norm, bar_width, label="Current Patient Sample", color="#E91E63" if primary_pred == "M" else "#AB47BC")
                ax_comp.bar(x_indices + bar_width, m_norm, bar_width, label="Malignant Population Avg", color="#D81B60", alpha=0.85)

                ax_comp.set_xticks(x_indices)
                ax_comp.set_xticklabels([f.replace("_mean", "") for f in comp_features], fontweight="600")
                ax_comp.set_ylabel("Ratio to Benign Mean", fontweight="600")
                ax_comp.set_title("Patient Measurements Relative to Benign Clinical Baseline (1.0 = Normal Benign)", fontsize=11, fontweight="bold", color="#880E4F")
                ax_comp.axhline(1.0, color="#888", linestyle="--", linewidth=0.8)
                ax_comp.legend(loc="upper right", framealpha=0.9)
                ax_comp.grid(True, linestyle=":", alpha=0.5)
                plt.tight_layout()
                st.pyplot(fig_comp)
                plt.close(fig_comp)

            st.markdown("""
            <div class="disclaimer-badge">
                <b>🎗️ Educational Viva Notice:</b> This diagnostic assessment is generated by an academic machine learning model for coursework demonstration.
                It is not certified or intended for clinical patient care or medical decision-making.
            </div>
            """, unsafe_allow_html=True)
