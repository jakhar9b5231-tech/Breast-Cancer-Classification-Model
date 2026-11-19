# ==============================================================================
# Script: create_screenshots.py
# Purpose: Executes each individual part of train_model.py, captures its output,
#          renders high-resolution terminal window screenshots of each part,
#          copies all generated plot screenshots, creates composite collages,
#          and packages everything into 'train_model_output/' and 'train_model_output.zip'.
# ==============================================================================

import os
import sys
import shutil
import zipfile
import textwrap
import subprocess
from PIL import Image, ImageDraw, ImageFont

# Directory setup
OUTPUT_DIR = "train_model_output"
PLOTS_DIR = "plots"
ZIP_NAME = "train_model_output.zip"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Font setup
windir = os.environ.get("WINDIR", "C:\\Windows")
font_reg = os.path.join(windir, "Fonts", "consola.ttf")
font_bold = os.path.join(windir, "Fonts", "consolab.ttf")

if not os.path.exists(font_reg):
    font_reg = "arial.ttf"
if not os.path.exists(font_bold):
    font_bold = "arialbd.ttf"


def render_terminal_window(title, command, text, output_filename, max_width=1150):
    """
    Renders an authentic, high-resolution dark-mode terminal window screenshot
    containing the executed command and its exact terminal stdout.
    """
    font = ImageFont.truetype(font_reg, 15)
    bold_font = ImageFont.truetype(font_bold, 15)
    title_font = ImageFont.truetype(font_reg, 13)

    raw_lines = [line.rstrip() for line in text.split("\n")]
    while raw_lines and not raw_lines[-1].strip():
        raw_lines.pop()

    # Wrap very long lines so the window remains at a clean proportional width
    wrapped_lines = []
    for line in raw_lines:
        if len(line) > 105 and not line.startswith("="):
            sub_lines = textwrap.wrap(line, width=105, subsequent_indent="    ")
            wrapped_lines.extend(sub_lines)
        else:
            wrapped_lines.append(line)

    line_height = 22
    pad_x = 28
    pad_top = 56
    pad_bottom = 25

    # Measure maximum text width
    dummy = Image.new("RGB", (10, 10))
    draw_dummy = ImageDraw.Draw(dummy)
    all_lines = [f"PS C:\\breast cancer wisconsin> {command}"] + wrapped_lines
    max_w = 0
    for line in all_lines:
        w = draw_dummy.textlength(line, font=font)
        if w > max_w:
            max_w = w

    img_w = max(1050, min(1200, int(max_w + pad_x * 2 + 35)))
    img_h = int(pad_top + len(all_lines) * line_height + pad_bottom)

    # Base background (modern terminal dark background)
    img = Image.new("RGB", (img_w, img_h), "#0d1117")
    draw = ImageDraw.Draw(img)

    # Terminal Title Bar
    draw.rectangle([0, 0, img_w, 42], fill="#161b22")
    draw.line([0, 42, img_w, 42], fill="#30363d", width=1)

    # Window Control Dots (macOS/modern terminal style)
    draw.ellipse([18, 15, 30, 27], fill="#ff5f56")  # Red
    draw.ellipse([38, 15, 50, 27], fill="#ffbd2e")  # Yellow
    draw.ellipse([58, 15, 70, 27], fill="#27c93f")  # Green

    # Title Text
    draw.text((85, 13), title, fill="#8b949e", font=title_font)

    # Command Prompt
    prompt_str = "PS C:\\breast cancer wisconsin> "
    draw.text((pad_x, pad_top), prompt_str, fill="#58a6ff", font=bold_font)
    p_len = draw_dummy.textlength(prompt_str, font=bold_font)
    draw.text((pad_x + p_len, pad_top), command, fill="#f0f6fc", font=font)

    # Terminal Output Lines
    y = pad_top + line_height + 4
    for line in wrapped_lines:
        color = "#c9d1d9"
        cur_font = font

        # Dynamic color highlights for terminal elements
        if line.startswith("="):
            color = "#e3b341"  # Golden amber for step borders
            cur_font = bold_font
        elif line.startswith("STEP") or "STEP " in line:
            color = "#79c0ff"  # Soft bright blue
            cur_font = bold_font
        elif line.startswith("---") or line.startswith("==="):
            color = "#58a6ff"
            cur_font = bold_font
        elif any(metric in line for metric in ["Accuracy:", "Precision:", "Recall:", "F1-Score:"]):
            color = "#56d364"  # Vibrant green for performance metrics
            cur_font = bold_font
        elif "Confusion Matrix" in line or "Classification Report" in line:
            color = "#79c0ff"
            cur_font = bold_font
        elif "Saved:" in line:
            color = "#a5d6ff"  # Cyan for file output confirmations
        elif "Selected Best Model:" in line or "TRAINING COMPLETED" in line:
            color = "#3fb950"  # Bright green
            cur_font = bold_font
        elif "DISCLAIMER" in line or "Clinical Relevance" in line or "Why False Negatives" in line:
            color = "#ffa657"  # Warm orange for disclaimers and clinical notes

        draw.text((pad_x, y), line, fill=color, font=cur_font)
        y += line_height

    # Outer border outline
    draw.rectangle([0, 0, img_w - 1, img_h - 1], outline="#30363d", width=1)

    out_path = os.path.join(OUTPUT_DIR, output_filename)
    img.save(out_path, dpi=(150, 150))
    print(f"  [+] Saved Screenshot: {out_path} ({img_w}x{img_h})")
    return out_path


def run_step(step_number, part_title):
    """Executes a single step of train_model.py and returns stdout."""
    cmd = [sys.executable, "train_model.py", f"--step={step_number}"]
    cmd_display = f"python train_model.py --step {step_number}"
    print(f"\n>>> Running Part {step_number}: {part_title} ...")
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
    if res.returncode != 0:
        print(f"Error in step {step_number}: {res.stderr}")
    return cmd_display, res.stdout


def create_eda_collage():
    """Generates a composite multi-panel overview of all 5 EDA plots."""
    plot_files = [
        "plots/target_distribution.png",
        "plots/correlation_heatmap.png",
        "plots/feature_histograms.png",
        "plots/feature_boxplots.png",
        "plots/scatter_plot.png"
    ]
    if all(os.path.exists(p) for p in plot_files):
        imgs = [Image.open(p) for p in plot_files]
        w = 1900
        h = 1750
        collage = Image.new("RGB", (w, h), "#ffffff")
        draw = ImageDraw.Draw(collage)

        # Title
        title_font = ImageFont.truetype(font_bold, 28)
        draw.text((30, 25), "Exploratory Data Analysis (EDA) - Visual Plots Summary", fill="#1e3d59", font=title_font)

        # Row 1: Target Distribution & Scatter Plot
        img0 = imgs[0].resize((750, 500))
        img4 = imgs[4].resize((1050, 500))
        collage.paste(img0, (40, 80))
        collage.paste(img4, (820, 80))

        # Row 2: Histograms & Boxplots
        img2 = imgs[2].resize((880, 520))
        img3 = imgs[3].resize((880, 520))
        collage.paste(img2, (40, 600))
        collage.paste(img3, (960, 600))

        # Row 3: Correlation Heatmap
        img1 = imgs[1].resize((880, 560))
        collage.paste(img1, (510, 1140))

        collage_path = os.path.join(OUTPUT_DIR, "03_part3_eda_all_plots_collage.png")
        collage.save(collage_path, dpi=(150, 150))
        print(f"  [+] Saved EDA Collage: {collage_path}")


def main():
    print("=" * 70)
    print("RUNNING EACH PART OF TRAIN_MODEL.PY AND GENERATING SCREENSHOTS")
    print("=" * 70)

    # 1. PART 1: DATA LOADING
    cmd1, out1 = run_step(1, "Data Loading")
    render_terminal_window(
        "Windows PowerShell - train_model.py (Part 1: Data Loading)",
        cmd1,
        out1,
        "01_part1_data_loading_terminal_output.png"
    )

    # 2. PART 2: DATA CLEANING
    cmd2, out2 = run_step(2, "Data Cleaning")
    render_terminal_window(
        "Windows PowerShell - train_model.py (Part 2: Data Cleaning)",
        cmd2,
        out2,
        "02_part2_data_cleaning_terminal_output.png"
    )

    # 3. PART 3: EXPLORATORY DATA ANALYSIS (EDA)
    cmd3, out3 = run_step(3, "Exploratory Data Analysis (EDA)")
    render_terminal_window(
        "Windows PowerShell - train_model.py (Part 3: Exploratory Data Analysis)",
        cmd3,
        out3,
        "03_part3_eda_terminal_output.png"
    )

    # Copy individual plots generated by Part 3
    eda_plots = [
        ("plots/target_distribution.png", "03_part3_plot_target_distribution.png"),
        ("plots/correlation_heatmap.png", "03_part3_plot_correlation_heatmap.png"),
        ("plots/feature_histograms.png", "03_part3_plot_feature_histograms.png"),
        ("plots/feature_boxplots.png", "03_part3_plot_feature_boxplots.png"),
        ("plots/scatter_plot.png", "03_part3_plot_scatter_plot.png"),
    ]
    for src, dst in eda_plots:
        if os.path.exists(src):
            dst_full = os.path.join(OUTPUT_DIR, dst)
            shutil.copy2(src, dst_full)
            print(f"  [+] Copied Plot Screenshot: {dst_full}")

    create_eda_collage()

    # 4. PART 4: DATA PREPROCESSING
    cmd4, out4 = run_step(4, "Data Preprocessing")
    render_terminal_window(
        "Windows PowerShell - train_model.py (Part 4: Data Preprocessing)",
        cmd4,
        out4,
        "04_part4_preprocessing_terminal_output.png"
    )

    # 5. PART 5: TRAINING MACHINE LEARNING MODELS
    cmd5, out5 = run_step(5, "Training ML Models")
    render_terminal_window(
        "Windows PowerShell - train_model.py (Part 5: ML Models Training)",
        cmd5,
        out5,
        "05_part5_models_training_terminal_output.png"
    )

    # 6. PART 6: MODEL EVALUATION
    cmd6, out6 = run_step(6, "Model Evaluation")
    render_terminal_window(
        "Windows PowerShell - train_model.py (Part 6: Model Evaluation)",
        cmd6,
        out6,
        "06_part6_model_evaluation_terminal_output.png"
    )
    if os.path.exists("plots/confusion_matrices.png"):
        dst_cm = os.path.join(OUTPUT_DIR, "06_part6_plot_confusion_matrices.png")
        shutil.copy2("plots/confusion_matrices.png", dst_cm)
        print(f"  [+] Copied Confusion Matrices Plot: {dst_cm}")

    # 7. PART 7: MODEL COMPARISON & SELECTION
    cmd7, out7 = run_step(7, "Model Comparison & Selection")
    render_terminal_window(
        "Windows PowerShell - train_model.py (Part 7: Model Comparison & Selection)",
        cmd7,
        out7,
        "07_part7_model_comparison_terminal_output.png"
    )
    if os.path.exists("plots/model_comparison.png"):
        dst_mc = os.path.join(OUTPUT_DIR, "07_part7_plot_model_accuracy_comparison.png")
        shutil.copy2("plots/model_comparison.png", dst_mc)
        print(f"  [+] Copied Model Comparison Bar Chart: {dst_mc}")

    # Create HTML gallery index for viewing all screenshots easily
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>train_model.py Execution Screenshots</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0f141c; color: #e6edf3; margin: 0; padding: 30px; }
        h1 { color: #58a6ff; border-bottom: 2px solid #30363d; padding-bottom: 12px; }
        h2 { color: #79c0ff; margin-top: 35px; border-left: 4px solid #58a6ff; padding-left: 10px; }
        p { color: #8b949e; line-height: 1.6; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
        img { max-width: 100%; height: auto; border-radius: 6px; border: 1px solid #30363d; margin-top: 10px; }
        .badge { display: inline-block; background: #238636; color: white; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-bottom: 8px; }
    </style>
</head>
<body>
    <h1>Breast Cancer Classification - train_model.py Output Screenshots</h1>
    <p>Complete record of terminal outputs and graphical visualization screenshots generated from executing each part of <code>train_model.py</code>.</p>

    <div class="card">
        <span class="badge">Part 1</span>
        <h2>Part 1: Data Loading Output</h2>
        <p>Loading <code>data/data.csv</code>, shape, head, columns, df.info(), statistical summary, nulls, and duplicate check.</p>
        <img src="01_part1_data_loading_terminal_output.png" alt="Part 1 Data Loading">
    </div>

    <div class="card">
        <span class="badge">Part 2</span>
        <h2>Part 2: Data Cleaning Output</h2>
        <p>Dropping <code>id</code> and <code>Unnamed: 32</code>, target distribution (Benign: 357, Malignant: 212), and feature separation.</p>
        <img src="02_part2_data_cleaning_terminal_output.png" alt="Part 2 Data Cleaning">
    </div>

    <div class="card">
        <span class="badge">Part 3</span>
        <h2>Part 3: Exploratory Data Analysis (EDA) Output</h2>
        <p>Terminal output and all 5 generated visualization plots.</p>
        <img src="03_part3_eda_terminal_output.png" alt="Part 3 EDA Terminal Output">
        <h3>Generated Plots:</h3>
        <img src="03_part3_eda_all_plots_collage.png" alt="EDA All Plots Collage">
    </div>

    <div class="card">
        <span class="badge">Part 4</span>
        <h2>Part 4: Data Preprocessing Output</h2>
        <p>80:20 Stratified train-test split and StandardScaler fitted strictly on training data.</p>
        <img src="04_part4_preprocessing_terminal_output.png" alt="Part 4 Preprocessing">
    </div>

    <div class="card">
        <span class="badge">Part 5</span>
        <h2>Part 5: Machine Learning Models Training Output</h2>
        <p>Training Gaussian Naive Bayes, Logistic Regression, and Support Vector Machine.</p>
        <img src="05_part5_models_training_terminal_output.png" alt="Part 5 ML Models Training">
    </div>

    <div class="card">
        <span class="badge">Part 6</span>
        <h2>Part 6: Model Evaluation Output</h2>
        <p>Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and Classification Reports for all three models.</p>
        <img src="06_part6_model_evaluation_terminal_output.png" alt="Part 6 Model Evaluation">
        <h3>Side-by-Side Confusion Matrices Plot:</h3>
        <img src="06_part6_plot_confusion_matrices.png" alt="Confusion Matrices Plot">
    </div>

    <div class="card">
        <span class="badge">Part 7</span>
        <h2>Part 7: Model Comparison & Selection Output</h2>
        <p>Model performance comparison table, Accuracy bar chart, and best model selection explanation.</p>
        <img src="07_part7_model_comparison_terminal_output.png" alt="Part 7 Model Comparison">
        <h3>Accuracy Comparison Bar Chart:</h3>
        <img src="07_part7_plot_model_accuracy_comparison.png" alt="Model Comparison Bar Chart">
    </div>
</body>
</html>
"""
    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  [+] Created HTML Gallery: {os.path.join(OUTPUT_DIR, 'index.html')}")

    # Compile all files into train_model_output.zip
    print("\n>>> Compiling all screenshots into zip archive: train_model_output.zip ...")
    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(OUTPUT_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, ".")
                zipf.write(file_path, arcname)
                print(f"  [archive] Added {arcname}")

    print("\n" + "=" * 70)
    print(f"SUCCESS! All screenshots compiled into directory: '{OUTPUT_DIR}/'")
    print(f"And zipped into archive: '{ZIP_NAME}'")
    print("=" * 70)


if __name__ == "__main__":
    main()
