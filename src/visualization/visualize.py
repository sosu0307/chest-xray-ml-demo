# src/visualization/visualize.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import math

# =========================
# Fixed Paths
# =========================
DATA_PATH = Path(r"D:\DS_ML\Repository\nov25_bds_int_covid1\data\processed\ML\features_merged_all.csv")
FIGURES_DIR = Path(r"D:\DS_ML\Repository\nov25_bds_int_covid1\reports\figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("DATA_PATH:", DATA_PATH)
print("FIGURES_DIR:", FIGURES_DIR)

# =========================
# Load Data
# =========================
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, sep=",")
    print("Data loaded:", df.shape)
    return df

df = load_data()
print("Columns:", df.columns.tolist())
print("First 5 values of size_kb:", df["size_kb"].head())
print("Data type:", df["size_kb"].dtype)

# =========================
# File Size Distribution Plot
# =========================
def plot_file_size_distribution(df):
    df_plot = pd.to_numeric(df["size_kb"], errors="coerce").dropna()
    plt.figure(figsize=(8,5))
    plt.hist(df_plot, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("File Size (KB)")
    plt.ylabel("Count")
    plt.title("File Size Distribution")
    plt.tight_layout()
    save_path = FIGURES_DIR / "file_size_distribution.png"
    plt.savefig(save_path)
    plt.close()
    print("File size plot saved:", save_path)

# =========================
# Class Distribution Plot
# =========================
def plot_class_distribution(df):
    diagnosis_cols = [
        "diagnosis_COVID",
        "diagnosis_Normal",
        "diagnosis_Lung_Opacity",
        "diagnosis_Viral_Pneumonia"
    ]
    class_names_map = {
        "diagnosis_COVID": "COVID",
        "diagnosis_Normal": "Normal",
        "diagnosis_Lung_Opacity": "Lung Opacity",
        "diagnosis_Viral_Pneumonia": "Viral Pneumonia"
    }
    class_counts = df[diagnosis_cols].sum()
    class_counts.index = [class_names_map[col] for col in class_counts.index]

    plt.figure(figsize=(8,5))
    colors = ["tab:red", "tab:blue", "tab:orange", "tab:green"]
    class_counts.plot(kind="bar", color=colors)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    save_path = FIGURES_DIR / "class_distribution.png"
    plt.savefig(save_path)
    plt.close()
    print("Class distribution plot saved:", save_path)

# =========================
# Feature Correlation Heatmap
# =========================
def plot_feature_correlation(df):
    feature_cols = ["COVID", "mean_intensity", "rms_contrast", "dark_pixel_ratio", "bright_pixel_ratio", "size_kb", "laplacian_variance", "entropy", "energy",
                    "lbp_mean", "lbp_std", "lbp_bin_0", "lbp_bin_1", "lbp_bin_2", "lbp_bin_3", "lbp_bin_4", "lbp_bin_5", "lbp_bin_6", "lbp_bin_7", 
                    "lbp_bin_8", "lbp_bin_9", "skew", "kurtosis", "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation", "glcm_entropy", "grad_mag_std",
                    "fft_high_freq_energy", "lung_area_ratio", "opacity_compactness", "opacity_eccentricity", "bbox_area_ratio"]
    corr = df[feature_cols].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    save_path = FIGURES_DIR / "feature_correlation_heatmap.png"
    plt.savefig(save_path)
    plt.close()
    print("Feature correlation heatmap saved:", save_path)

# =========================
# Pairplot / Scatter Matrix
# =========================
def plot_pairplot(df):
    feature_cols = ["mean_intensity", "rms_contrast", "dark_pixel_ratio", "bright_pixel_ratio", "size_kb", "entropy", "lbp_std", "lbp_bin_2", "lbp_bin_5", "lbp_bin_7",
                    "skew", "kurtosis", "glcm_contrast", "glcm_entropy", "grad_mag_std"]
    plt.figure(figsize=(8,6))
    sns.pairplot(df, vars=feature_cols, hue='label', palette='tab10', diag_kind='kde')
    plt.suptitle("Pairplot of Features by Class", y=1.02)
    save_path = FIGURES_DIR / "pairplot_features.png"
    plt.savefig(save_path)
    plt.close()
    print("Pairplot saved:", save_path)

# =========================
# Boxplots per Feature by Class
# =========================
def plot_boxplots(df):
    feature_cols = ["mean_intensity", "rms_contrast", "dark_pixel_ratio", "bright_pixel_ratio", "size_kb"]

    for col in feature_cols:
        plt.figure(figsize=(8,5))
        sns.boxplot(x='label', y=col, data=df, palette='tab10')
        plt.title(f"{col} by Class")
        plt.xlabel("Class")
        plt.ylabel(col)
        plt.xticks(rotation=15)
        plt.tight_layout()
        save_path = FIGURES_DIR / f"boxplot_{col}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Boxplot for {col} saved:", save_path)

def plot_boxplots_subplots(df):
    feature_cols = ["entropy", "lbp_std", "lbp_bin_2", "lbp_bin_5", "lbp_bin_7", "skew", "kurtosis", "glcm_contrast", "glcm_entropy", "grad_mag_std"]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        sns.boxplot(x=df['label'], y=df[col], ax=axes[i])
        axes[i].set_title(f'{col}')
        axes[i].set_xlabel('')

    plt.tight_layout()
    plt.show()

# =========================
# Violin plots per Feature by Class
# =========================
def plot_violin_plot_subplots(df):
    feature_cols = ["entropy", "lbp_std", "lbp_bin_2", "lbp_bin_5", "lbp_bin_7", "skew", "kurtosis", "glcm_contrast", "glcm_entropy", "grad_mag_std"]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        sns.violinplot(x=df['label'], y=df[col], ax=axes[i], inner='box')
        axes[i].set_title(f'{col}')
        axes[i].set_xlabel('')

    plt.tight_layout()
    plt.show()

# =========================
# Scatterplots
# =========================
def plot_scatterplots(df):
    sns.set(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatterplot: dark_pixel_ratio vs mean_intensity
    sns.scatterplot(
        data=df,
        x='dark_pixel_ratio',
        y='mean_intensity',
        ax=axes[0],
        color='blue',
        alpha=0.6
    )
    axes[0].set_title('Dark Pixel Ratio vs Mean Intensity')
    axes[0].set_xlabel('Dark Pixel Ratio')
    axes[0].set_ylabel('Mean Intensity')

    # Scatterplot: dark_pixel_ratio vs rms_contrast
    sns.scatterplot(
        data=df,
        x='dark_pixel_ratio',
        y='rms_contrast',
        ax=axes[1],
        color='green',
        alpha=0.6
    )
    axes[1].set_title('Dark Pixel Ratio vs RMS Contrast')
    axes[1].set_xlabel('Dark Pixel Ratio')
    axes[1].set_ylabel('RMS Contrast')

    plt.tight_layout()
    save_path = FIGURES_DIR / "scatterplots_dark_ratio.png"
    plt.savefig(save_path)
    plt.close()
    print("Scatterplots saved:", save_path)

def plot_single_scatter(df):
    plt.figsize=(16, 6)

    sns.scatterplot(
        data=df,
        x='grad_mag_std',
        y='glcm_contrast',
        hue='COVID',
        palette={0: 'tab:blue', 1: 'tab:orange'},
        alpha=0.6
    )
    plt.title('glcm_contrast vs grad_mag_std')
    plt.xlabel('grad_mag_std')
    plt.ylabel('glcm_contrast')
    plt.tight_layout()
    plt.show()

def plot_scatter_grid(df, variables, y_var, hue_col, cols_per_row=4):
    # Calculate how many rows we need
    num_plots = len(variables)
    num_rows = math.ceil(num_plots / cols_per_row)
    
    # Set up the figure and flatten so we can loop with a single index
    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(cols_per_row * 5, num_rows * 4))
    axes = axes.flatten()
    
    # Define color palette
    custom_palette = {df[hue_col].unique()[0]: 'tab:orange', 
                      df[hue_col].unique()[1]: 'tab:blue'}

    # Loop through the variables and plot
    for i, var in enumerate(variables):
        sns.scatterplot(
            data=df,
            x=var, 
            y=y_var, 
            hue=hue_col,
            ax=axes[i],
            alpha=0.6,
            palette=custom_palette,
            edgecolor='w',
            linewidth=0.5
        )
        
        axes[i].set_title(f'{y_var} vs {var}', fontsize=12)

    # Hide any empty subplots (if num_plots is not a perfect multiple of cols_per_row)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# =========================
# Run all plots
# =========================
if __name__ == "__main__":
    print("__main__ is running")
    #plot_file_size_distribution(df)
    #plot_class_distribution(df)
    #plot_feature_correlation(df)
    #plot_pairplot(df)
    #plot_boxplots(df)
    #plot_boxplots_subplots(df)
    #plot_violin_plot_subplots(df)
    #plot_scatterplots(df)
    plot_single_scatter(df)
    #plot_scatter_grid(df, ["mean_intensity", "rms_contrast", "dark_pixel_ratio", "skew"], 'kurtosis', hue_col='COVID', cols_per_row=4)
    print("All plots generated successfully")
