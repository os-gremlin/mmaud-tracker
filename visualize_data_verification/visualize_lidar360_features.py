import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
from datetime import datetime
import shutil

plt.ioff()

# ============================================================
# USER PATHS
# ============================================================

FEATURE_DIR = r"C:\swarm\MMOAD\Calibration Data\results\lidar_360_feature_set"
RESULT_ROOT = r"C:\swarm\MMOAD\Calibration Data\results"

FEATURE_PATH = os.path.join(FEATURE_DIR, "feature_val.npy")
LABEL_PATH   = os.path.join(FEATURE_DIR, "label_val.npy")

# ============================================================
# CREATE TIMESTAMPED OUTPUT FOLDER
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(RESULT_ROOT, timestamp)
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[INFO] Export directory: {OUT_DIR}")

# ============================================================
# LOAD DATA
# ============================================================

if not os.path.isfile(FEATURE_PATH):
    raise FileNotFoundError(f"feature_val.npy not found at: {FEATURE_PATH}")
if not os.path.isfile(LABEL_PATH):
    raise FileNotFoundError(f"label_val.npy not found at: {LABEL_PATH}")

features = np.load(FEATURE_PATH)   # (N, 20, 9)
labels   = np.load(LABEL_PATH)     # (N,)

print(f"[INFO] Loaded features shape: {features.shape}")
print(f"[INFO] Loaded labels shape:   {labels.shape}")

if features.shape[0] != labels.shape[0]:
    raise ValueError("Number of feature samples and labels do not match!")

N, T, F = features.shape
print(f"[INFO] N_samples={N}, T={T}, F={F}")

# ============================================================
# DATASET SUMMARY FIGURE
# ============================================================

def save_dataset_summary(features, labels, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Label distribution
    ax = axes[0]
    unique, counts = np.unique(labels, return_counts=True)
    ax.bar(unique, counts, tick_label=[str(u) for u in unique])
    ax.set_title("Label Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")

    # Simple feature value histogram (flattened)
    ax = axes[1]
    flat = features.reshape(-1)
    ax.hist(flat, bins=50)
    ax.set_title("Global Feature Value Histogram")
    ax.set_xlabel("Feature value")
    ax.set_ylabel("Frequency")

    fig.suptitle("Lidar360 Feature Dataset Summary", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(out_dir, "dataset_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved dataset summary: {out_path}")

save_dataset_summary(features, labels, OUT_DIR)

# ============================================================
# SELECT EXAMPLE SAMPLES (ONE POSITIVE, ONE NEGATIVE)
# ============================================================

pos_indices = np.where(labels == 1)[0]
neg_indices = np.where(labels == 0)[0]

if pos_indices.size == 0:
    raise ValueError("No positive (label=1) samples found!")
if neg_indices.size == 0:
    raise ValueError("No negative (label=0) samples found!")

# For reproducibility, just pick the first ones
pos_idx = int(pos_indices[0])
neg_idx = int(neg_indices[0])

print(f"[INFO] Using positive sample index: {pos_idx}")
print(f"[INFO] Using negative sample index: {neg_idx}")

feat_pos = features[pos_idx]  # (T, F)
feat_neg = features[neg_idx]  # (T, F)

# ============================================================
# HELPER: SAVE HEATMAP AND LINE PLOTS AS PNG
# ============================================================

def save_sample_pngs(feat_tf, label, idx, out_dir, prefix):
    """
    feat_tf: (T, F)
    label: int
    idx: sample index
    prefix: 'pos' or 'neg'
    """
    T, F = feat_tf.shape
    times = np.arange(T)

    # 1) Heatmap of (T x F)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(feat_tf, aspect='auto', origin='lower')
    ax.set_title(f"{prefix.upper()} sample {idx} (label={label})\nFeature Heatmap (T x F)")
    ax.set_xlabel("Feature index (0..F-1)")
    ax.set_ylabel("Time index (0..T-1)")
    fig.colorbar(im, ax=ax, label="Feature value")
    out_path_hm = os.path.join(out_dir, f"{prefix}_sample_{idx}_heatmap.png")
    fig.tight_layout()
    fig.savefig(out_path_hm, dpi=150)
    plt.close(fig)

    # 2) Line plots for each feature across time
    fig, ax = plt.subplots(figsize=(7, 4))
    for f_idx in range(F):
        ax.plot(times, feat_tf[:, f_idx], label=f"f{f_idx}", alpha=0.7)
    ax.set_title(f"{prefix.upper()} sample {idx} (label={label})\nFeature Time-Series")
    ax.set_xlabel("Time index (0..T-1)")
    ax.set_ylabel("Feature value")
    ax.grid(True, alpha=0.3)
    # Optionally, show legend outside
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    out_path_line = os.path.join(out_dir, f"{prefix}_sample_{idx}_lines.png")
    fig.tight_layout()
    fig.savefig(out_path_line, dpi=150)
    plt.close(fig)

    print(f"[INFO] Saved {prefix} sample PNGs:")
    print(f"       Heatmap: {out_path_hm}")
    print(f"       Lines:   {out_path_line}")

# Save PNGs for both positive and negative samples
save_sample_pngs(feat_pos, 1, pos_idx, OUT_DIR, "pos")
save_sample_pngs(feat_neg, 0, neg_idx, OUT_DIR, "neg")

# ============================================================
# ANIMATION HELPERS
# We create three animations *for the positive sample*:
#   A) Heatmap animation
#   B) Line-plot animation
#   C) Combined heatmap + lines
# ============================================================

# Decide on writer (ffmpeg if available, otherwise Pillow writer / gif)
use_ffmpeg = shutil.which("ffmpeg") is not None
if use_ffmpeg:
    WriterClass = FFMpegWriter
    ext = "mp4"
    print("[INFO] ffmpeg found: using MP4 output.")
else:
    WriterClass = PillowWriter
    ext = "gif"
    print("[WARN] ffmpeg not found: using GIF output instead of MP4.")

def animate_heatmap(feat_tf, out_dir, idx, prefix="pos"):
    """
    A) Animate the (T, F) heatmap row-by-row:
       each frame shows a 1 x F heatmap for time t
    """
    T, F = feat_tf.shape
    fig, ax = plt.subplots(figsize=(5, 3))

    def init():
        ax.clear()
        ax.set_title(f"{prefix.upper()} sample {idx} - Heatmap Animation")
        ax.set_xlabel("Feature index (0..F-1)")
        ax.set_ylabel("Time slice")
        return []

    def update(frame):
        ax.clear()
        row = feat_tf[frame:frame+1, :]  # (1, F)
        im = ax.imshow(row, aspect='auto', origin='lower', vmin=feat_tf.min(), vmax=feat_tf.max())
        ax.set_title(f"{prefix.upper()} sample {idx} - Heatmap (time={frame})")
        ax.set_xlabel("Feature index (0..F-1)")
        ax.set_ylabel("Time slice (1 row)")
        return [im]

    writer = WriterClass(fps=5)
    out_path = os.path.join(out_dir, f"{prefix}_sample_{idx}_anim_heatmap.{ext}")
    with writer.saving(fig, out_path, dpi=150):
        for f in range(T):
            update(f)
            writer.grab_frame()
    plt.close(fig)
    print(f"[INFO] Saved heatmap animation: {out_path}")

def animate_lines(feat_tf, out_dir, idx, prefix="pos"):
    """
    B) Animate line plots for all features across time:
       each frame shows data up to the current time index.
    """
    T, F = feat_tf.shape
    times = np.arange(T)

    fig, ax = plt.subplots(figsize=(6, 4))
    lines = []
    for f_idx in range(F):
        (line,) = ax.plot([], [], label=f"f{f_idx}", alpha=0.7)
        lines.append(line)

    ax.set_xlim(0, T - 1)
    # Use global min/max for stable y-axis
    ymin = np.min(feat_tf)
    ymax = np.max(feat_tf)
    pad = 0.05 * (ymax - ymin + 1e-6)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Feature value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    def update(frame):
        ax.set_title(f"{prefix.upper()} sample {idx} - Feature Lines (t=0..{frame})")
        for f_idx, line in enumerate(lines):
            line.set_data(times[:frame+1], feat_tf[:frame+1, f_idx])
        return lines

    writer = WriterClass(fps=5)
    out_path = os.path.join(out_dir, f"{prefix}_sample_{idx}_anim_lines.{ext}")
    with writer.saving(fig, out_path, dpi=150):
        for f in range(T):
            update(f)
            writer.grab_frame()
    plt.close(fig)
    print(f"[INFO] Saved line-plot animation: {out_path}")

def animate_combined(feat_tf, out_dir, idx, prefix="pos"):
    """
    C) Combined animation:
       Left: 1 x F heatmap for current time
       Right: line plots up to current time
    """
    T, F = feat_tf.shape
    times = np.arange(T)

    fig, (ax_hm, ax_ln) = plt.subplots(1, 2, figsize=(10, 4))

    # Precompute min/max for consistent color scale
    vmin = np.min(feat_tf)
    vmax = np.max(feat_tf)

    # Setup line plot
    lines = []
    for f_idx in range(F):
        (line,) = ax_ln.plot([], [], label=f"f{f_idx}", alpha=0.7)
        lines.append(line)

    ax_ln.set_xlim(0, T - 1)
    ymin = np.min(feat_tf)
    ymax = np.max(feat_tf)
    pad = 0.05 * (ymax - ymin + 1e-6)
    ax_ln.set_ylim(ymin - pad, ymax + pad)
    ax_ln.set_xlabel("Time index")
    ax_ln.set_ylabel("Feature value")
    ax_ln.grid(True, alpha=0.3)
    ax_ln.legend(loc="upper right", fontsize=8, ncol=2)

    def update(frame):
        # Heatmap (left)
        ax_hm.clear()
        row = feat_tf[frame:frame+1, :]  # (1, F)
        im = ax_hm.imshow(row, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        ax_hm.set_title(f"Heatmap (time={frame})")
        ax_hm.set_xlabel("Feature index")
        ax_hm.set_ylabel("Time slice (1 row)")

        # Lines (right)
        ax_ln.set_title(f"Lines (t=0..{frame})")
        for f_idx, line in enumerate(lines):
            line.set_data(times[:frame+1], feat_tf[:frame+1, f_idx])

        fig.suptitle(f"{prefix.upper()} sample {idx} - Combined View", fontsize=14)
        return [im] + lines

    writer = WriterClass(fps=5)
    out_path = os.path.join(out_dir, f"{prefix}_sample_{idx}_anim_combined.{ext}")
    with writer.saving(fig, out_path, dpi=150):
        for f in range(T):
            update(f)
            writer.grab_frame()
    plt.close(fig)
    print(f"[INFO] Saved combined animation: {out_path}")

# ============================================================
# RUN ALL THREE ANIMATIONS FOR THE POSITIVE SAMPLE
# ============================================================

print("[INFO] Creating animations for POSITIVE sample...")
animate_heatmap(feat_pos, OUT_DIR, pos_idx, prefix="pos")
animate_lines(feat_pos, OUT_DIR, pos_idx, prefix="pos")
animate_combined(feat_pos, OUT_DIR, pos_idx, prefix="pos")

print("[DONE] Visualization complete.")
print(f"[DONE] All outputs saved in: {OUT_DIR}")
