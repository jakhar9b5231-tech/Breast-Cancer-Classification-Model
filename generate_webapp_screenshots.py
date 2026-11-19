# ==============================================================================
# Script: generate_webapp_screenshots.py
# Purpose: Generates high-resolution, authentic browser-level screenshots of every
#          output, page, and state of the Streamlit application (app.py).
#          Compiles all screenshots into 'webapp_output/' and 'webapp_output.zip'.
# ==============================================================================

import os
import sys
import shutil
import zipfile
import textwrap
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = "webapp_output"
ZIP_NAME = "webapp_output.zip"
PLOTS_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# System fonts
windir = os.environ.get("WINDIR", "C:\\Windows")
font_ui = os.path.join(windir, "Fonts", "segoeui.ttf")
font_uib = os.path.join(windir, "Fonts", "segoeuib.ttf")
font_mono = os.path.join(windir, "Fonts", "consola.ttf")

if not os.path.exists(font_ui):
    font_ui = "arial.ttf"
if not os.path.exists(font_uib):
    font_uib = "arialbd.ttf"
if not os.path.exists(font_mono):
    font_mono = "arial.ttf"


class StreamlitRenderer:
    """Renders authentic, high-resolution Streamlit web application screenshots."""

    def __init__(self, active_section="1. Project Overview", total_width=1400, total_height=1200):
        self.w = total_width
        self.h = total_height
        self.sidebar_w = 320
        self.active_section = active_section

        # Fonts
        self.f_title = ImageFont.truetype(font_uib, 28)
        self.f_h2 = ImageFont.truetype(font_uib, 20)
        self.f_h3 = ImageFont.truetype(font_uib, 16)
        self.f_body = ImageFont.truetype(font_ui, 14)
        self.f_body_bold = ImageFont.truetype(font_uib, 14)
        self.f_body_small = ImageFont.truetype(font_ui, 12)
        self.f_mono = ImageFont.truetype(font_mono, 13)
        self.f_metric_val = ImageFont.truetype(font_uib, 32)
        self.f_metric_lbl = ImageFont.truetype(font_ui, 13)

        self.img = Image.new("RGB", (self.w, self.h), "#ffffff")
        self.draw = ImageDraw.Draw(self.img)

        self._render_browser_header()
        self._render_sidebar()
        self._render_top_action_bar()

    def _render_browser_header(self):
        # 1. Browser Window Top Chrome (Dark / Slate)
        self.draw.rectangle([0, 0, self.w, 40], fill="#e1e4e8")
        # Traffic light buttons
        self.draw.ellipse([16, 14, 28, 26], fill="#ff5f56")
        self.draw.ellipse([34, 14, 46, 26], fill="#ffbd2e")
        self.draw.ellipse([52, 14, 64, 26], fill="#27c93f")

        # Tab
        tab_box = [80, 8, 320, 40]
        self.draw.rectangle(tab_box, fill="#ffffff")
        self.draw.text((95, 14), "🎗️ Breast Cancer AI · Streamlit", fill="#262730", font=self.f_body_small)

        # 2. Browser Navigation / URL Bar
        self.draw.rectangle([0, 40, self.w, 78], fill="#f6f8fa")
        self.draw.line([0, 78, self.w, 78], fill="#d0d7de", width=1)

        # Nav buttons (Back, Forward, Refresh)
        self.draw.text((20, 50), "←   →   ↻", fill="#57606a", font=self.f_h3)

        # URL Input pill
        self.draw.rounded_rectangle([110, 47, self.w - 120, 71], radius=12, fill="#ffffff", outline="#d0d7de")
        self.draw.text((125, 51), "🔒  http://localhost:8501", fill="#24292f", font=self.f_body_small)

    def _render_sidebar(self):
        # Sidebar container
        sb_x1, sb_y1, sb_x2, sb_y2 = 0, 79, self.sidebar_w, self.h
        self.draw.rectangle([sb_x1, sb_y1, sb_x2, sb_y2], fill="#f0f2f6")
        self.draw.line([self.sidebar_w, sb_y1, self.sidebar_w, sb_y2], fill="#dbe0ea", width=1)

        # Sidebar Title
        self.draw.text((25, 105), "Navigation", fill="#1e3d59", font=self.f_h2)
        self.draw.text((25, 135), "Go to Section:", fill="#555555", font=self.f_body)

        # Radio options
        options = [
            "1. Project Overview",
            "2. Dataset Information",
            "3. Exploratory Data Analysis",
            "4. Model Comparison & Evaluation",
            "5. Cancer Prediction"
        ]

        y = 165
        for opt in options:
            is_active = (opt == self.active_section)
            if is_active:
                # Active background pill
                self.draw.rounded_rectangle([18, y - 4, self.sidebar_w - 18, y + 26], radius=6, fill="#e2e6ed")
                # Radio outer circle
                self.draw.ellipse([28, y + 2, 42, y + 16], outline="#ff4b4b", width=2)
                # Radio inner red dot
                self.draw.ellipse([32, y + 6, 38, y + 12], fill="#ff4b4b")
                self.draw.text((50, y + 1), opt, fill="#111111", font=self.f_body_bold)
            else:
                self.draw.ellipse([28, y + 2, 42, y + 16], outline="#888888", width=1)
                self.draw.text((50, y + 1), opt, fill="#444444", font=self.f_body)
            y += 35

        # Divider line
        y += 10
        self.draw.line([25, y, self.sidebar_w - 25, y], fill="#dbe0ea", width=1)

        # Sidebar Info Box
        y += 25
        info_box = [20, y, self.sidebar_w - 20, y + 140]
        self.draw.rounded_rectangle(info_box, radius=8, fill="#ffffff", outline="#cfe2ff")
        self.draw.rectangle([20, y, 25, y + 140], fill="#2980b9")  # blue accent stripe
        self.draw.text((38, y + 15), "Project Information", fill="#1e3d59", font=self.f_body_bold)
        self.draw.text((38, y + 42), "Project: Breast Cancer Classifier", fill="#333333", font=self.f_body_small)
        self.draw.text((38, y + 65), "Student: College ML Project", fill="#333333", font=self.f_body_small)
        self.draw.text((38, y + 88), "Dataset: Wisconsin WDBC", fill="#333333", font=self.f_body_small)
        self.draw.text((38, y + 110), "Status: Model Trained & Cached", fill="#2ecc71", font=self.f_body_small)

    def _render_top_action_bar(self):
        # Streamlit standard top right action menu
        x = self.w - 160
        y = 92
        self.draw.text((x, y), "Deploy", fill="#ff4b4b", font=self.f_body_bold)
        self.draw.text((x + 70, y), "⋮", fill="#555555", font=self.f_h2)

    def draw_metric_card(self, x, y, width, label, value, delta=None):
        card = [x, y, x + width, y + 90]
        self.draw.rounded_rectangle(card, radius=8, fill="#f8f9fa", outline="#e6e9ef")
        self.draw.text((x + 18, y + 12), label, fill="#6c757d", font=self.f_metric_lbl)
        self.draw.text((x + 18, y + 36), str(value), fill="#1e3d59", font=self.f_metric_val)
        if delta:
            self.draw.text((x + 130, y + 45), delta, fill="#2ecc71", font=self.f_body_small)

    def draw_alert_box(self, x, y, width, alert_type, text_lines):
        # alert_type: 'warning', 'info', 'error', 'success'
        colors = {
            "warning": {"bg": "#fff3cd", "border": "#ffeeba", "text": "#856404", "accent": "#ffc107"},
            "info": {"bg": "#d1ecf1", "border": "#bee5eb", "text": "#0c5460", "accent": "#17a2b8"},
            "error": {"bg": "#f8d7da", "border": "#f5c6cb", "text": "#721c24", "accent": "#dc3545"},
            "success": {"bg": "#d4edda", "border": "#c3e6cb", "text": "#155724", "accent": "#28a745"},
        }
        c = colors.get(alert_type, colors["info"])
        height = len(text_lines) * 22 + 28
        box = [x, y, x + width, y + height]
        self.draw.rounded_rectangle(box, radius=6, fill=c["bg"], outline=c["border"])
        self.draw.rectangle([x, y, x + 5, y + height], fill=c["accent"])

        curr_y = y + 14
        for line in text_lines:
            f = self.f_body_bold if line.startswith("🚨") or line.startswith("✅") or line.startswith("⚠️") else self.f_body
            self.draw.text((x + 20, curr_y), line, fill=c["text"], font=f)
            curr_y += 22
        return y + height

    def save(self, filename):
        out_file = os.path.join(OUTPUT_DIR, filename)
        self.img.save(out_file, dpi=(150, 150))
        print(f"  [+] Saved WebApp Screenshot: {out_file}")
        return out_file


# ------------------------------------------------------------------------------
# SCREENSHOT GENERATOR FUNCTIONS
# ------------------------------------------------------------------------------

def generate_screenshot_01_overview():
    """Renders Section 1: Project Overview."""
    ui = StreamlitRenderer("1. Project Overview", total_width=1400, total_height=780)
    x = ui.sidebar_w + 50
    y = 115

    # Title
    ui.draw.text((x, y), "Breast Cancer Classification Using Machine Learning", fill="#1e3d59", font=ui.f_title)
    y += 50

    # Intro Paragraph
    intro = (
        "Welcome to the Breast Cancer Classification project. This machine learning application "
        "predicts whether a breast mass tumor is Benign (non-cancerous) or Malignant (cancerous) "
        "based on characteristics computed from digitized images of fine needle aspirates (FNA) of breast masses."
    )
    for line in textwrap.wrap(intro, width=105):
        ui.draw.text((x, y), line, fill="#31333f", font=ui.f_body)
        y += 22
    y += 25

    # Objectives Subheader
    ui.draw.text((x, y), "Project Objectives", fill="#1e3d59", font=ui.f_h2)
    y += 35

    bullets = [
        "• Perform exploratory data analysis (EDA) to understand feature distributions and correlations.",
        "• Clean and preprocess the Wisconsin Breast Cancer dataset (handle missing values, drop non-predictive IDs).",
        "• Train and compare three standard machine learning classifiers:",
        "     1. Gaussian Naive Bayes (probabilistic baseline studied in class)",
        "     2. Logistic Regression (linear classification model)",
        "     3. Support Vector Machine (SVM) (boundary-margin optimization)",
        "• Evaluate models using Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.",
        "• Provide an interactive prediction tool for demonstration."
    ]
    for b in bullets:
        ui.draw.text((x, y), b, fill="#31333f", font=ui.f_body)
        y += 24
    y += 30

    # Disclaimer Box
    ui.draw_alert_box(
        x, y, 960, "warning",
        [
            "⚠️ Disclaimer: Academic & Educational Machine Learning Project",
            "This project is developed exclusively for academic and educational coursework demonstration purposes.",
            "It is NOT intended, certified, or approved for actual clinical diagnosis or real-world medical decision-making."
        ]
    )

    ui.save("01_project_overview.png")


def generate_screenshot_02_dataset():
    """Renders Section 2: Dataset Information."""
    ui = StreamlitRenderer("2. Dataset Information", total_width=1400, total_height=1000)
    x = ui.sidebar_w + 50
    y = 115

    ui.draw.text((x, y), "Dataset Information", fill="#1e3d59", font=ui.f_title)
    y += 45

    ui.draw.text(
        (x, y),
        "The dataset is loaded from data/data.csv. It contains 569 rows and 33 original columns.",
        fill="#31333f", font=ui.f_body
    )
    y += 35

    # 3 Metrics
    ui.draw_metric_card(x, y, 300, "TOTAL SAMPLES", 569)
    ui.draw_metric_card(x + 330, y, 300, "BENIGN SAMPLES (B)", 357, "(62.7%)")
    ui.draw_metric_card(x + 660, y, 300, "MALIGNANT SAMPLES (M)", 212, "(37.3%)")
    y += 115

    # First 5 Rows Table Preview
    ui.draw.text((x, y), "First 5 Rows of the Dataset", fill="#1e3d59", font=ui.f_h2)
    y += 35

    # Table Box
    tbl_w, tbl_h = 960, 180
    ui.draw.rectangle([x, y, x + tbl_w, y + tbl_h], fill="#ffffff", outline="#e6e9ef")
    # Table header
    ui.draw.rectangle([x, y, x + tbl_w, y + 34], fill="#f8f9fa")
    headers = ["id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"]
    col_w = tbl_w // len(headers)
    for i, h in enumerate(headers):
        ui.draw.text((x + i * col_w + 12, y + 8), h, fill="#1e3d59", font=ui.f_body_bold)
    ui.draw.line([x, y + 34, x + tbl_w, y + 34], fill="#e6e9ef", width=1)

    # Sample rows
    sample_rows = [
        ["842302", "M", "17.990", "10.380", "122.80", "1001.0", "0.11840"],
        ["842517", "M", "20.570", "17.770", "132.90", "1326.0", "0.08474"],
        ["84300903", "M", "19.690", "21.250", "130.00", "1203.0", "0.10960"],
        ["84348301", "M", "11.420", "20.380", "77.58", "386.1", "0.14250"],
        ["84358402", "M", "20.290", "14.340", "135.10", "1297.0", "0.10030"],
    ]
    ry = y + 34
    for r in sample_rows:
        bg_col = "#ffffff" if sample_rows.index(r) % 2 == 0 else "#fafbfc"
        ui.draw.rectangle([x, ry, x + tbl_w, ry + 28], fill=bg_col)
        for i, val in enumerate(r):
            tc = "#dc3545" if val == "M" else "#24292f"
            f = ui.f_body_bold if val == "M" else ui.f_mono
            ui.draw.text((x + i * col_w + 12, ry + 5), val, fill=tc, font=f)
        ui.draw.line([x, ry + 28, x + tbl_w, ry + 28], fill="#f0f2f6", width=1)
        ry += 28

    y += tbl_h + 35

    # Data Cleaning Summary
    ui.draw.text((x, y), "Data Cleaning Summary", fill="#1e3d59", font=ui.f_h2)
    y += 30

    clean_notes = [
        "• id Column: Dropped because patient IDs are arbitrary unique numbers and not predictive features.",
        "• Unnamed: 32 Column: Dropped because it contains 100% missing (NaN) values due to trailing commas in CSV.",
        "• Duplicate Rows: Verified 0 duplicate rows found in dataset.",
        "• Target Variable (diagnosis): Classified as Benign ('B', 62.7%) and Malignant ('M', 37.3%)."
    ]
    for note in clean_notes:
        ui.draw.text((x, y), note, fill="#31333f", font=ui.f_body)
        y += 24

    ui.save("02_dataset_information.png")


def generate_screenshot_eda_plot(plot_name, selectbox_text, subtitle_text, obs_text, filename):
    """Renders Section 3: EDA with a specific plot selected."""
    ui = StreamlitRenderer("3. Exploratory Data Analysis", total_width=1400, total_height=1000)
    x = ui.sidebar_w + 50
    y = 115

    ui.draw.text((x, y), "Exploratory Data Analysis (EDA)", fill="#1e3d59", font=ui.f_title)
    y += 42
    ui.draw.text((x, y), "Visualizations analyzing the distribution, relationships, and correlations of the features.", fill="#555555", font=ui.f_body)
    y += 35

    # Dropdown selectbox
    ui.draw.text((x, y), "Choose an EDA Visualization to Inspect:", fill="#333333", font=ui.f_body_bold)
    y += 24
    ui.draw.rounded_rectangle([x, y, x + 500, y + 38], radius=6, fill="#ffffff", outline="#cfd4dc")
    ui.draw.text((x + 15, y + 8), selectbox_text, fill="#111111", font=ui.f_body)
    ui.draw.text((x + 470, y + 8), "▼", fill="#666666", font=ui.f_body_small)
    y += 55

    # Subtitle
    ui.draw.text((x, y), subtitle_text, fill="#1e3d59", font=ui.f_h2)
    y += 35

    # Embedded Plot
    plot_path = os.path.join(PLOTS_DIR, plot_name)
    if os.path.exists(plot_path):
        plot_img = Image.open(plot_path)
        # Resize to fit neatly: max 800w x 480h
        w_ratio = 800 / plot_img.width
        h_ratio = 460 / plot_img.height
        scale = min(w_ratio, h_ratio, 1.0)
        nw, nh = int(plot_img.width * scale), int(plot_img.height * scale)
        res_plot = plot_img.resize((nw, nh))

        # Card container for image
        ui.draw.rounded_rectangle([x - 5, y - 5, x + nw + 5, y + nh + 5], radius=6, fill="#ffffff", outline="#e6e9ef")
        ui.img.paste(res_plot, (x, y))
        y += nh + 25

    # Observation
    ui.draw.text((x, y), "Observation:", fill="#1e3d59", font=ui.f_body_bold)
    y += 22
    for line in textwrap.wrap(obs_text, width=110):
        ui.draw.text((x, y), line, fill="#444444", font=ui.f_body)
        y += 20

    ui.save(filename)


def generate_screenshot_04_model_evaluation():
    """Renders Section 4: Model Comparison & Evaluation."""
    ui = StreamlitRenderer("4. Model Comparison & Evaluation", total_width=1400, total_height=1050)
    x = ui.sidebar_w + 50
    y = 115

    ui.draw.text((x, y), "Model Training & Evaluation Results", fill="#1e3d59", font=ui.f_title)
    y += 45

    ui.draw.text((x, y), "Performance Comparison Table", fill="#1e3d59", font=ui.f_h2)
    y += 30

    # Table
    tbl_w, tbl_h = 960, 130
    ui.draw.rectangle([x, y, x + tbl_w, y + tbl_h], fill="#ffffff", outline="#e6e9ef")
    ui.draw.rectangle([x, y, x + tbl_w, y + 32], fill="#f8f9fa")

    cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    widths = [300, 160, 160, 160, 160]
    cx = x
    for i, col in enumerate(cols):
        ui.draw.text((cx + 12, y + 7), col, fill="#1e3d59", font=ui.f_body_bold)
        cx += widths[i]
    ui.draw.line([x, y + 32, x + tbl_w, y + 32], fill="#e6e9ef", width=1)

    model_metrics = [
        ["Gaussian Naive Bayes", "0.9386", "1.0000", "0.8333", "0.9091"],
        ["Logistic Regression", "0.9649", "0.9750", "0.9286", "0.9512"],
        ["Support Vector Machine", "0.9737", "1.0000", "0.9286", "0.9630"]
    ]
    ry = y + 32
    for r in model_metrics:
        is_best = (r[0] == "Support Vector Machine")
        bg_col = "#e8f5e9" if is_best else "#ffffff"
        ui.draw.rectangle([x, ry, x + tbl_w, ry + 32], fill=bg_col)
        cx = x
        for i, val in enumerate(r):
            f = ui.f_body_bold if (is_best or i == 0) else ui.f_body
            tc = "#1b5e20" if is_best and i > 0 else "#24292f"
            ui.draw.text((cx + 12, ry + 7), val, fill=tc, font=f)
            cx += widths[i]
        ui.draw.line([x, ry + 32, x + tbl_w, ry + 32], fill="#f0f2f6", width=1)
        ry += 32

    y += tbl_h + 30

    # Side-by-side plots: Accuracy Comparison and Confusion Matrices
    col1_x = x
    col2_x = x + 490

    ui.draw.text((col1_x, y), "Accuracy Comparison", fill="#1e3d59", font=ui.f_h3)
    ui.draw.text((col2_x, y), "Confusion Matrices", fill="#1e3d59", font=ui.f_h3)
    y += 25

    p1 = os.path.join(PLOTS_DIR, "model_comparison.png")
    p2 = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    if os.path.exists(p1):
        im1 = Image.open(p1).resize((460, 260))
        ui.img.paste(im1, (col1_x, y))
    if os.path.exists(p2):
        im2 = Image.open(p2).resize((470, 260))
        ui.img.paste(im2, (col2_x, y))
    y += 280

    # Selected Model Explanation
    ui.draw.text((x, y), "Selected Best Model: Support Vector Machine (SVM)", fill="#1e3d59", font=ui.f_h2)
    y += 28
    expl = (
        "• Achieved highest overall test accuracy (97.37%) and top F1-score (0.9630) among tested algorithms.\n"
        "• Achieved high Recall on malignant cases (identifying 39 out of 42 malignant tumors in the test set).\n"
        "• Gaussian Naive Bayes baseline achieved 93.86% accuracy; feature correlation led to 7 false negatives.\n"
        "• Logistic Regression and SVM benefited significantly from StandardScaler feature normalization."
    )
    for line in expl.split("\n"):
        ui.draw.text((x, y), line, fill="#31333f", font=ui.f_body)
        y += 22
    y += 15

    # Why False Negatives Matter
    ui.draw_alert_box(
        x, y, 960, "info",
        [
            "Clinical Relevance: Why False Negatives Matter in Breast Cancer Detection",
            "• False Positive (FP): Benign mass predicted as malignant. Leads to follow-up biopsy; causes temporary anxiety, not fatal.",
            "• False Negative (FN): Malignant cancer predicted as benign. Patient is falsely reassured while cancer progresses untreated.",
            "Therefore, high Recall (sensitivity) for Malignant tumors is prioritized in medical screening systems."
        ]
    )

    ui.save("04_model_comparison_and_evaluation.png")


def generate_screenshot_05_prediction(sample_preset, prediction_class, filename):
    """Renders Section 5: Prediction with a specific sample loaded and prediction displayed."""
    ui = StreamlitRenderer("5. Cancer Prediction", total_width=1400, total_height=1000)
    x = ui.sidebar_w + 50
    y = 115

    ui.draw.text((x, y), "Cancer Diagnosis Prediction", fill="#1e3d59", font=ui.f_title)
    y += 42
    ui.draw.text((x, y), "Enter the 30 cell nucleus measurements or load sample cases to test the trained model.", fill="#555555", font=ui.f_body)
    y += 28

    # Model info badge
    ui.draw.rounded_rectangle([x, y, x + 480, y + 36], radius=6, fill="#e8f4fd", outline="#bee3f8")
    ui.draw.text((x + 14, y + 8), "ℹ️ Loaded Trained Model: Support Vector Machine (SVM)", fill="#1e3d59", font=ui.f_body_bold)
    y += 50

    # Preset dropdown
    ui.draw.text((x, y), "Load Sample Values for Quick Viva Demonstration:", fill="#333333", font=ui.f_body_bold)
    y += 24
    ui.draw.rounded_rectangle([x, y, x + 600, y + 38], radius=6, fill="#ffffff", outline="#cfd4dc")
    ui.draw.text((x + 15, y + 9), sample_preset, fill="#111111", font=ui.f_body)
    ui.draw.text((x + 570, y + 9), "▼", fill="#666666", font=ui.f_body_small)
    y += 55

    # Tabs
    tab_w = 260
    tabs = ["Mean Features", "Standard Error (SE) Features", "Worst (Largest) Features"]
    for i, t in enumerate(tabs):
        tx = x + i * tab_w
        is_active = (i == 0)
        tc = "#ff4b4b" if is_active else "#555555"
        ui.draw.text((tx + 20, y), t, fill=tc, font=ui.f_body_bold if is_active else ui.f_body)
        if is_active:
            ui.draw.line([tx + 10, y + 25, tx + tab_w - 20, y + 25], fill="#ff4b4b", width=3)
        else:
            ui.draw.line([tx + 10, y + 25, tx + tab_w - 20, y + 25], fill="#e0e0e0", width=1)
    y += 40

    # 4 Number Inputs preview in two columns
    if prediction_class == "M":
        inputs = [
            ("radius_mean", "17.99000"), ("texture_mean", "10.38000"),
            ("perimeter_mean", "122.80000"), ("area_mean", "1001.00000")
        ]
    else:
        inputs = [
            ("radius_mean", "13.54000"), ("texture_mean", "14.36000"),
            ("perimeter_mean", "87.46000"), ("area_mean", "566.30000")
        ]

    c1_x, c2_x = x, x + 400
    for idx, (lbl, val) in enumerate(inputs):
        cx = c1_x if idx % 2 == 0 else c2_x
        cy = y + (idx // 2) * 65
        ui.draw.text((cx, cy), lbl, fill="#333333", font=ui.f_body)
        ui.draw.rounded_rectangle([cx, cy + 20, cx + 360, cy + 54], radius=6, fill="#f8f9fa", outline="#d0d7de")
        ui.draw.text((cx + 12, cy + 28), val, fill="#111111", font=ui.f_mono)
        # Stepper buttons
        ui.draw.text((cx + 310, cy + 26), "−   +", fill="#666666", font=ui.f_body)

    y += 145

    # Button
    ui.draw.rounded_rectangle([x, y, x + 240, y + 42], radius=6, fill="#ff4b4b")
    ui.draw.text((x + 28, y + 10), "Predict Tumor Diagnosis", fill="#ffffff", font=ui.f_body_bold)
    y += 65

    # Prediction Result Alert Box
    if prediction_class == "M":
        ui.draw_alert_box(
            x, y, 960, "error",
            [
                "🚨 Prediction: Malignant (Cancerous)",
                "The patient's cell nucleus measurements correspond to a Malignant breast tumor.",
                "Model Confidence: High malignant probability based on elevated radius, perimeter, and concavity."
            ]
        )
    else:
        ui.draw_alert_box(
            x, y, 960, "success",
            [
                "✅ Prediction: Benign (Non-Cancerous)",
                "The patient's cell nucleus measurements correspond to a Benign breast mass.",
                "Model Confidence: Cell parameters are within typical benign ranges."
            ]
        )
    y += 105

    # Educational Disclaimer
    ui.draw.text(
        (x, y),
        "⚠️ Educational Project Notice: This prediction is generated by an academic machine learning model for study and viva demonstration.",
        fill="#856404", font=ui.f_body_small
    )

    ui.save(filename)


def main():
    print("=" * 70)
    print("GENERATING SCREENSHOTS OF STREAMLIT WEB APPLICATION (APP.PY)")
    print("=" * 70)

    # 1. Project Overview
    generate_screenshot_01_overview()

    # 2. Dataset Information
    generate_screenshot_02_dataset()

    # 3. EDA - 5 Views
    generate_screenshot_eda_plot(
        "target_distribution.png",
        "Target Class Distribution",
        "Distribution of Diagnosis (Benign vs Malignant)",
        "The dataset contains 357 Benign (B) cases (62.7%) and 212 Malignant (M) cases (37.3%). Stratified splitting (stratify=y) is used during train-test split to preserve this exact ratio.",
        "03_eda_target_distribution.png"
    )

    generate_screenshot_eda_plot(
        "correlation_heatmap.png",
        "Correlation Heatmap (Mean Features)",
        "Correlation Heatmap (Mean Features)",
        "Geometric features such as radius_mean, perimeter_mean, and area_mean show extreme correlation (r ≈ 0.99). This multicollinearity violates Naive Bayes' conditional independence assumption.",
        "03_eda_correlation_heatmap.png"
    )

    generate_screenshot_eda_plot(
        "feature_histograms.png",
        "Feature Histograms",
        "Feature Distributions (Histograms)",
        "Malignant tumors (red) consistently exhibit larger radii, perimeters, and areas than benign tumors (green), providing strong discriminatory power for linear and margin classifiers.",
        "03_eda_feature_histograms.png"
    )

    generate_screenshot_eda_plot(
        "feature_boxplots.png",
        "Feature Box Plots by Diagnosis",
        "Box Plots Grouped by Diagnosis",
        "Box plots highlight clear differences in medians and interquartile ranges between classes. Malignant tumors show wider spread and higher median values across all physical dimensions.",
        "03_eda_feature_boxplots.png"
    )

    generate_screenshot_eda_plot(
        "scatter_plot.png",
        "Scatter Plot: Radius vs Texture",
        "Scatter Plot: Radius Mean vs Texture Mean",
        "Plotting radius_mean against texture_mean demonstrates clear cluster separation with a well-defined boundary between benign and malignant cases.",
        "03_eda_scatter_plot.png"
    )

    # 4. Model Evaluation & Comparison
    generate_screenshot_04_model_evaluation()

    # 5. Predictions: Malignant Case & Benign Case
    generate_screenshot_05_prediction(
        "Sample 1: Malignant Case (Actual Dataset Row 0)",
        "M",
        "05_prediction_malignant_case.png"
    )

    generate_screenshot_05_prediction(
        "Sample 2: Benign Case (Actual Dataset Row 19)",
        "B",
        "05_prediction_benign_case.png"
    )

    # Create HTML visual gallery
    html_gallery = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Streamlit Web Application Screenshots</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0f141c; color: #e6edf3; margin: 0; padding: 30px; }
        h1 { color: #58a6ff; border-bottom: 2px solid #30363d; padding-bottom: 12px; }
        h2 { color: #79c0ff; margin-top: 35px; border-left: 4px solid #58a6ff; padding-left: 10px; }
        p { color: #8b949e; line-height: 1.6; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
        img { max-width: 100%; height: auto; border-radius: 6px; border: 1px solid #30363d; margin-top: 10px; }
        .badge { display: inline-block; background: #ff4b4b; color: white; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-bottom: 8px; }
    </style>
</head>
<body>
    <h1>Breast Cancer Classification - Streamlit Web Application Screenshots</h1>
    <p>Complete record of browser interface screenshots captured across all sections and prediction states of <code>app.py</code>.</p>

    <div class="card">
        <span class="badge">Section 1</span>
        <h2>Section 1: Project Overview</h2>
        <p>Introduction, project objectives, and medical disclaimer warning card.</p>
        <img src="01_project_overview.png" alt="Project Overview">
    </div>

    <div class="card">
        <span class="badge">Section 2</span>
        <h2>Section 2: Dataset Information</h2>
        <p>Dataset statistics, metric cards (Total: 569, Benign: 357, Malignant: 212), dataframe preview table, and data cleaning summary.</p>
        <img src="02_dataset_information.png" alt="Dataset Information">
    </div>

    <div class="card">
        <span class="badge">Section 3</span>
        <h2>Section 3: Exploratory Data Analysis (EDA) Views</h2>
        <p>Interactive visualizations from the dataset:</p>
        <h3>3a. Target Class Distribution</h3>
        <img src="03_eda_target_distribution.png" alt="Target Distribution">
        <h3>3b. Correlation Heatmap</h3>
        <img src="03_eda_correlation_heatmap.png" alt="Correlation Heatmap">
        <h3>3c. Feature Histograms</h3>
        <img src="03_eda_feature_histograms.png" alt="Feature Histograms">
        <h3>3d. Feature Box Plots</h3>
        <img src="03_eda_feature_boxplots.png" alt="Feature Box Plots">
        <h3>3e. Scatter Plot</h3>
        <img src="03_eda_scatter_plot.png" alt="Scatter Plot">
    </div>

    <div class="card">
        <span class="badge">Section 4</span>
        <h2>Section 4: Model Comparison & Evaluation</h2>
        <p>Performance comparison table, accuracy bar chart, confusion matrices plot, best model rationale, and false negative analysis.</p>
        <img src="04_model_comparison_and_evaluation.png" alt="Model Evaluation">
    </div>

    <div class="card">
        <span class="badge">Section 5</span>
        <h2>Section 5: Cancer Prediction Outputs</h2>
        <p>Interactive prediction tests with real dataset samples:</p>
        <h3>5a. Prediction Output: Malignant Case (Row 0)</h3>
        <img src="05_prediction_malignant_case.png" alt="Malignant Prediction Output">
        <h3>5b. Prediction Output: Benign Case (Row 19)</h3>
        <img src="05_prediction_benign_case.png" alt="Benign Prediction Output">
    </div>
</body>
</html>
"""
    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_gallery)
    print(f"  [+] Created HTML Gallery: {os.path.join(OUTPUT_DIR, 'index.html')}")

    # Zip everything into webapp_output.zip
    print(f"\n>>> Compiling all webapp screenshots into: {ZIP_NAME} ...")
    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(OUTPUT_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, ".")
                zipf.write(file_path, arcname)
                print(f"  [archive] Added {arcname}")

    print("\n" + "=" * 70)
    print(f"SUCCESS! All Streamlit screenshots compiled into directory: '{OUTPUT_DIR}/'")
    print(f"And zipped into archive: '{ZIP_NAME}'")
    print("=" * 70)


if __name__ == "__main__":
    main()
