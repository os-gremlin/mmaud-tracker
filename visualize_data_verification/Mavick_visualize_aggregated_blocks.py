import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from datetime import datetime
from sklearn.cluster import DBSCAN
import csv

# ======================================================================
# USER PATHS (MAVIC2 DATASET)
# ======================================================================

ROOT = r"C:/swarm/MMOAD/MAVIC2"          # MAVIC2 root folder
RESULT_ROOT = os.path.join(ROOT, "results")

# Name to use in overlays / filenames (no seq subfolder here)
SEQ = "MAVIC2"

# ======================================================================
# CREATE TIMESTAMPED EXPORT FOLDER
# ======================================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(RESULT_ROOT, timestamp)
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Export directory: {OUT_DIR}")

# ======================================================================
# LOAD FOLDERS
# ======================================================================

# MAVIC2 is a flat dataset: lidar_360/ and ground_truth/ directly under ROOT
lidar_path = os.path.join(ROOT, "lidar_360")
gt_path    = os.path.join(ROOT, "ground_truth")

for p, name in [(lidar_path, "LiDAR360"),
                (gt_path, "Ground Truth")]:
    if not os.path.isdir(p):
        raise FileNotFoundError(f"{name} folder missing: {p}")

# ======================================================================
# LOAD TIMESTAMP–POINT DICTIONARIES
# ======================================================================

def load_timestamp_dict(folder):
    out = {}
    for f in os.listdir(folder):
        if f.endswith(".npy"):
            ts = float(os.path.splitext(f)[0])
            out[ts] = os.path.join(folder, f)
    return dict(sorted(out.items(), key=lambda x: x[0]))

lidar_dict = load_timestamp_dict(lidar_path)
gt_dict    = load_timestamp_dict(gt_path)

lidar_ts = np.array(sorted(lidar_dict.keys()))
gt_ts    = np.array(sorted(gt_dict.keys()))

if lidar_ts.size == 0:
    raise RuntimeError("No LiDAR .npy files found in lidar_360 folder.")
if gt_ts.size == 0:
    raise RuntimeError("No ground truth .npy files found in ground_truth folder.")

# ======================================================================
# HELPERS: GET NEAREST GT
# ======================================================================

def get_gt_for_timestamp(t):
    idx = np.abs(gt_ts - t).argmin()
    arr = np.load(gt_dict[gt_ts[idx]])
    return np.array(arr, dtype=float).reshape(3,)

# ======================================================================
# LOAD FULL GT TRAJECTORY (FOR PANEL 3 / 4)
# ======================================================================

gt_all_ts = np.array(sorted(gt_dict.keys()))
gt_all_xyz = []
for ts in gt_all_ts:
    arr = np.load(gt_dict[ts]).reshape(3,)
    gt_all_xyz.append(arr)
gt_all_xyz = np.vstack(gt_all_xyz) if len(gt_all_xyz) else np.empty((0, 3))

# ======================================================================
# ACCUMULATE 20-FRAME LiDAR360 BLOCKS
# ======================================================================

BLOCK_SIZE = 20
blocks = []

acc_points = []
count = 0

for t in lidar_ts:
    pts = np.load(lidar_dict[t])

    # Filter out all-zero points (common in Livox)
    mask = np.any(pts != 0, axis=1)
    pts = pts[mask][:, :3]   # xyz only

    # frame index for potential temporal coloring later
    frame_idx = np.full((len(pts), 1), count)
    pts_with_idx = np.hstack([frame_idx, pts])

    if count == 0:
        block_start_ts = t

    acc_points.append(pts_with_idx)
    count += 1

    if count == BLOCK_SIZE:
        acc_points = np.vstack(acc_points)
        blocks.append((block_start_ts, acc_points))
        acc_points = []
        count = 0

print(f"Found {len(blocks)} aggregated 20-frame blocks.")

if not blocks:
    raise RuntimeError("No full 20-frame blocks could be formed from MAVIC2 LiDAR.")

# ======================================================================
# GLOBAL AXIS LIMITS — across all blocks and GT
# ======================================================================

all_xyz = []

# From LiDAR blocks
for _, pts in blocks:
    xyz = pts[:, 1:4]
    if xyz.size:
        all_xyz.append(xyz)

# From GT
if gt_all_xyz.size:
    all_xyz.append(gt_all_xyz)

if len(all_xyz):
    all_xyz = np.vstack(all_xyz)
    Xmin, Xmax = float(np.min(all_xyz[:, 0])), float(np.max(all_xyz[:, 0]))
    Ymin, Ymax = float(np.min(all_xyz[:, 1])), float(np.max(all_xyz[:, 1]))
    Zmin, Zmax = float(np.min(all_xyz[:, 2])), float(np.max(all_xyz[:, 2]))
else:
    Xmin = Ymin = Zmin = -10.0
    Xmax = Ymax = Zmax = 10.0

pad = 1.0
Xmin -= pad; Xmax += pad
Ymin -= pad; Ymax += pad
Zmin -= pad; Zmax += pad

# ======================================================================
# SETUP FIGURE (4 PANELS)
# ======================================================================

fig = plt.figure(figsize=(24, 6))

ax1 = fig.add_subplot(141, projection="3d")  # LiDAR360 + GT + centroid
ax2 = fig.add_subplot(142, projection="3d")  # Cluster-only
ax3 = fig.add_subplot(143, projection="3d")  # GT trajectory
ax4 = fig.add_subplot(144, projection="3d")  # Centroid vs GT trajectory

cluster_colors = [
    "red", "green", "blue", "purple",
    "orange", "cyan", "magenta", "yellow"
]

# History storage for Panel 4 & extra plots
centroid_history = []
gt_history = []

# ======================================================================
# ANIMATION WRITER
# ======================================================================

out_video = os.path.join(OUT_DIR, f"{SEQ}_4panel_mid360_gt_centroid.mp4")
writer = FFMpegWriter(fps=1)
print("Saving MP4 to:", out_video)

# ======================================================================
# CSV OUTPUT (with centroid + GT positions)
# ======================================================================

csv_path = os.path.join(OUT_DIR, f"{SEQ}_4panel_block_stats.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "block_idx", "t_start",
    "num_points", "num_clusters", "num_noise",
    "gt_cluster_id", "gt_error",
    "gt_x", "gt_y", "gt_z",
    "centroid_x", "centroid_y", "centroid_z"
])

# ======================================================================
# ANIMATION LOOP
# ======================================================================

with writer.saving(fig, out_video, 200):

    for block_idx, (t_start, pts) in enumerate(blocks):

        # -------------------------------------------------------------
        # Reset axes and limits for all panels
        # -------------------------------------------------------------
        for ax in (ax1, ax2, ax3, ax4):
            ax.cla()
            ax.set_xlim(Xmin, Xmax)
            ax.set_ylim(Ymin, Ymax)
            ax.set_zlim(Zmin, Zmax)

        # -------------------------------------------------------------
        # PANEL 1: Full LiDAR360 block + DBSCAN + GT
        # -------------------------------------------------------------
        xyz = pts[:, 1:4]  # drop frame index column
        db = DBSCAN(eps=1, min_samples=10).fit(xyz)
        labels = db.labels_

        cluster_ids = sorted(set(labels))
        real_clusters = [c for c in cluster_ids if c != -1]
        num_noise = int(np.sum(labels == -1))

        # GT for this block
        gt_xyz = get_gt_for_timestamp(t_start)

        # Find the cluster whose centroid is closest to GT
        gt_cluster_id = None
        min_dist = 1e9
        centroid = None

        for cid in real_clusters:
            cpts = xyz[labels == cid]
            if cpts.size == 0:
                continue
            c_mean = np.mean(cpts, axis=0)
            d = np.linalg.norm(c_mean - gt_xyz)
            if d < min_dist:
                min_dist = d
                centroid = c_mean
                gt_cluster_id = cid

        # Plot clusters in Panel 1
        plotted_any = False
        for cid in cluster_ids:
            cpts = xyz[labels == cid]
            if cpts.size == 0:
                continue
            plotted_any = True
            if cid == -1:
                ax1.scatter(cpts[:, 0], cpts[:, 1], cpts[:, 2],
                            s=3, c="lightgray", alpha=0.3)
            else:
                col = cluster_colors[cid % len(cluster_colors)]
                ax1.scatter(cpts[:, 0], cpts[:, 1], cpts[:, 2],
                            s=3, c=col, alpha=0.7)

        if not plotted_any:
            ax1.text2D(0.35, 0.5, "EMPTY BLOCK",
                       transform=ax1.transAxes,
                       fontsize=18, color="red")

        # GT marker
        ax1.scatter(gt_xyz[0], gt_xyz[1], gt_xyz[2],
                    s=120, c="red", marker="X")

        # Highlight centroid of GT-matched cluster
        if centroid is not None:
            ax1.scatter(centroid[0], centroid[1], centroid[2],
                        s=200, c="yellow", marker="o", edgecolors="black")

        ax1.set_title("LiDAR360 Block (20 frames)")
        ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

        # -------------------------------------------------------------
        # PANEL 2: Cluster-only view
        # -------------------------------------------------------------
        for cid in cluster_ids:
            cpts = xyz[labels == cid]
            if cpts.size == 0:
                continue
            if cid == -1:
                ax2.scatter(cpts[:, 0], cpts[:, 1], cpts[:, 2],
                            s=3, c="lightgray", alpha=0.3)
            else:
                col = cluster_colors[cid % len(cluster_colors)]
                ax2.scatter(cpts[:, 0], cpts[:, 1], cpts[:, 2],
                            s=3, c=col, alpha=0.7)

        ax2.set_title("Cluster-Only View")
        ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

        # -------------------------------------------------------------
        # PANEL 3: Ground Truth Trajectory
        # -------------------------------------------------------------
        if gt_all_xyz.size:
            hist_idx = np.where(gt_all_ts <= t_start)[0]
            if hist_idx.size:
                hist_xyz = gt_all_xyz[hist_idx]
                ax3.plot(hist_xyz[:, 0], hist_xyz[:, 1], hist_xyz[:, 2],
                         linewidth=2, alpha=0.7, color="green")

        ax3.scatter(gt_xyz[0], gt_xyz[1], gt_xyz[2],
                    s=120, c="red", marker="X")
        ax3.set_title("Ground Truth Trajectory")
        ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")

        # -------------------------------------------------------------
        # PANEL 4: Centroid vs Ground Truth Trajectory
        # -------------------------------------------------------------
        if centroid is not None:
            centroid_history.append(centroid)
            gt_history.append(gt_xyz)

        # Draw historical centroid path
        if len(centroid_history) > 1:
            ch = np.vstack(centroid_history)
            ax4.plot(ch[:, 0], ch[:, 1], ch[:, 2],
                     color="yellow", linewidth=2, label="Centroid Path")

        # Draw historical GT path
        if len(gt_history) > 1:
            gh = np.vstack(gt_history)
            ax4.plot(gh[:, 0], gh[:, 1], gh[:, 2],
                     color="green", linewidth=2, label="GT Path")

        # Current centroid and GT points + vector
        if centroid is not None:
            ax4.scatter(centroid[0], centroid[1], centroid[2],
                        s=120, c="yellow", marker="o", edgecolors="black")
            ax4.scatter(gt_xyz[0], gt_xyz[1], gt_xyz[2],
                        s=120, c="red", marker="X")

            # Line connecting centroid → GT
            ax4.plot([centroid[0], gt_xyz[0]],
                     [centroid[1], gt_xyz[1]],
                     [centroid[2], gt_xyz[2]],
                     color="black", linewidth=2)

        ax4.set_title("Centroid vs Ground Truth Trajectory")
        ax4.set_xlabel("X"); ax4.set_ylabel("Y"); ax4.set_zlabel("Z")

        # -------------------------------------------------------------
        # OVERLAY INFO + CSV
        # -------------------------------------------------------------
        overlay = (
            f"Seq: {SEQ}\n"
            f"Block: {block_idx+1}/{len(blocks)}\n"
            f"Start TS: {t_start:.3f}\n"
            f"L360 pts: {xyz.shape[0]}\n"
            f"Clusters: {len(real_clusters)}\n"
            f"Noise pts: {num_noise}\n"
            f"GT cluster: {gt_cluster_id}\n"
            f"GT error: {min_dist:.3f} m"
        )
        ax1.text2D(0.02, 0.98, overlay,
                   transform=ax1.transAxes,
                   fontsize=9, verticalalignment="top")

        # Write CSV row with GT + centroid positions
        csv_writer.writerow([
            block_idx + 1,
            t_start,
            int(xyz.shape[0]),
            len(real_clusters),
            num_noise,
            gt_cluster_id if gt_cluster_id is not None else -1,
            float(min_dist),
            float(gt_xyz[0]), float(gt_xyz[1]), float(gt_xyz[2]),
            float(centroid[0]) if centroid is not None else np.nan,
            float(centroid[1]) if centroid is not None else np.nan,
            float(centroid[2]) if centroid is not None else np.nan
        ])

        # Grab frame for MP4
        writer.grab_frame()

csv_file.close()
plt.close(fig)

print("\n✓ 4-panel MAVIC2 LiDAR360 + GT + Centroid animation saved.")
print("MP4:", out_video)
print("CSV:", csv_path)

# ======================================================================
# ADDITIONAL PLOTS (4 FIGURES)
# Saved to OUT_DIR
# ======================================================================

centroid_hist = np.vstack(centroid_history) if len(centroid_history) else np.empty((0, 3))
gt_hist       = np.vstack(gt_history)       if len(gt_history)       else np.empty((0, 3))

# ============================
# 1) Centroid drift over time
# ============================
if centroid_hist.size and gt_hist.size:
    plt.figure(figsize=(10, 4))
    errors = [np.linalg.norm(centroid_hist[i] - gt_hist[i])
              for i in range(len(gt_hist))]
    plt.plot(range(1, len(errors) + 1), errors, marker='o')
    plt.xlabel("Block Index")
    plt.ylabel("Centroid–GT Error (m)")
    plt.title("Centroid Error per Aggregated Block")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{SEQ}_drift_over_time.png"))
    plt.close()

# ============================
# 2) 3D Trajectory comparison
# ============================
if gt_hist.size or centroid_hist.size:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    if gt_hist.size:
        ax.plot(gt_hist[:, 0], gt_hist[:, 1], gt_hist[:, 2],
                color='green', label='GT Trajectory', linewidth=2)
    if centroid_hist.size:
        ax.plot(centroid_hist[:, 0], centroid_hist[:, 1], centroid_hist[:, 2],
                color='yellow', label='Centroid Trajectory', linewidth=2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("3D Trajectory: Centroid vs Ground Truth")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{SEQ}_3D_centroid_vs_gt.png"))
    plt.close()

# ============================
# 3) BEV (top-down) XY drift path
# ============================
if gt_hist.size or centroid_hist.size:
    plt.figure(figsize=(8, 6))
    if gt_hist.size:
        plt.plot(gt_hist[:, 0], gt_hist[:, 1],
                 color='green', label='GT', linewidth=2)
    if centroid_hist.size:
        plt.plot(centroid_hist[:, 0], centroid_hist[:, 1],
                 color='yellow', label='Centroid', linewidth=2)

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Top-Down Trajectory (BEV)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{SEQ}_BEV_centroid_vs_gt.png"))
    plt.close()

# ============================
# 4) 2D Drift Arrows (vector errors)
# ============================
if gt_hist.size and centroid_hist.size:
    plt.figure(figsize=(8, 6))
    for i in range(len(gt_hist)):
        gx, gy = gt_hist[i, 0], gt_hist[i, 1]
        cx, cy = centroid_hist[i, 0], centroid_hist[i, 1]
        plt.arrow(gx, gy, cx - gx, cy - gy,
                  head_width=0.3, length_includes_head=True,
                  color="red", alpha=0.7)

    plt.scatter(gt_hist[:, 0], gt_hist[:, 1], c='green', label='GT')
    plt.scatter(centroid_hist[:, 0], centroid_hist[:, 1], c='yellow', label='Centroid')

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("2D Drift Vectors (GT → Centroid)")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{SEQ}_drift_vectors.png"))
    plt.close()

print("Extra plots saved to:")
print(" - drift_over_time.png")
print(" - 3D_centroid_vs_gt.png")
print(" - BEV_centroid_vs_gt.png")
print(" - drift_vectors.png")
