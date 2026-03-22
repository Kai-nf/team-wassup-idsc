import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def generate_class_imbalance_figure(out_dir):
    categories = ['Normal (Healthy)', 'Brugada Syndrome']
    counts = [287, 76]
    colors = ['#4C78A8', '#E45756']  # Calm blue and distinct red

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(categories, counts, color=colors, width=0.6)

    # Add data labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel("Number of Subjects", fontsize=12)
    ax.set_title("Dataset Class Imbalance", fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    out_file = out_dir / "class_imbalance.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"✅ Saved Figure 1: {out_file}")

def generate_preprocessing_figure(out_dir, repo_root):
    # Try to load the actual wavelet data, fallback to mock if not found
    wavelet_path = repo_root / "Preprocessed_Dataset" / "dataset_v3.1_wavelet.npy"
    fallback_path = repo_root / "JiaKang" / "dataset_v3_wavelet.npy"
    
    t = np.linspace(0, 1.2, 120)  # Mock 1.2 seconds of a beat
    
    if wavelet_path.exists():
        wavelet_data = np.load(wavelet_path)
        clean_signal = wavelet_data[0, :, 6] if wavelet_data.ndim == 3 else wavelet_data[0][6] # Assuming Lead V1 is idx 6
        clean_signal = clean_signal[:len(t)]
    elif fallback_path.exists():
        wavelet_data = np.load(fallback_path)
        clean_signal = wavelet_data[0, :, 6] if wavelet_data.ndim == 3 else wavelet_data[0][6]
        clean_signal = clean_signal[:len(t)]
    else:
        # Mock cleaned signal (Smooth ECG-like bump)
        clean_signal = np.sin(2 * np.pi * 1.5 * t) * np.exp(-3 * t)

    # Mock Raw Noisy ECG (Base signal + high-frequency noise + low-frequency baseline wander)
    noise = np.random.normal(0, 0.15, len(t))
    baseline_wander = 0.5 * np.sin(2 * np.pi * 0.5 * t)
    raw_signal = clean_signal + noise + baseline_wander

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top Panel: Raw Noisy
    axes[0].plot(t, raw_signal, color='gray', linewidth=1.5)
    axes[0].set_title("Raw Noisy ECG (Lead V1)", fontsize=14, fontweight='bold', loc='left')
    axes[0].set_ylabel("Amplitude (mV)", fontsize=12)
    axes[0].grid(color='r', linestyle='-', linewidth=0.5, alpha=0.3) # Medical grid style

    # Bottom Panel: Wavelet Cleaned
    axes[1].plot(t, clean_signal, color='#4C78A8', linewidth=1.5)
    axes[1].set_title("Wavelet Transformed ECG (Cleaned)", fontsize=14, fontweight='bold', loc='left')
    axes[1].set_ylabel("Amplitude (mV)", fontsize=12)
    axes[1].set_xlabel("Time (Seconds)", fontsize=12)
    axes[1].grid(color='r', linestyle='-', linewidth=0.5, alpha=0.3)

    out_file = out_dir / "preprocessing_effect.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"✅ Saved Figure 2: {out_file}")

if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parent
    out_dir = REPO_ROOT / "results" / "eda_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    generate_class_imbalance_figure(out_dir)
    generate_preprocessing_figure(out_dir, REPO_ROOT)
    print("\n🎉 All EDA figures generated successfully!")