import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from datetime import datetime
from sklearn.cluster import DBSCAN
import csv

# ======================================================================
# USER PATHS
# ======================================================================

SEQ = "seq0102"   # <-- change to any seqXXXX
ROOT = r"C:/swarm/MMOAD/Calibration Data/train"
RESULT_ROOT = r"C:/swarm/MMOAD/Calibration Data/results"

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

seq_path   = os.path.join(ROOT, SEQ)
lidar_path = os.path.join(seq_path, "lidar_360")
gt_path    = os.path.join(seq_path, "ground_truth")

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

# ======================================================================
# HELPERS: GET NEAREST GT
# ======================================================================

def get_gt_for_timestamp(t):
    idx = np.abs(gt_ts - t).argmin()
    arr = np.load(gt_dict[gt_ts[idx]])
    return np.array(arr, dtype=float).reshape(3,)

# ======================================================================
# LOAD FULL GT TRAJECTORY (FOR PANEL 3)
# ======================================================================

gt_all_ts = np.array(sorted(gt_dict.keys()))
gt_all_xyz = []
for ts in gt_all_ts:
    arr = np.load(gt_dict[ts]).reshape(3,)
    gt_all_xyz.append(arr)
gt_all_xyz = np.vstack(gt_all_xyz) if len(gt_all_xyz) else np.empty((0,3))

# ======================================================================
# ACCUMULATE 20-FRAME LiDAR360 BLOCKS
# ======================================================================

BLOCK_SIZE = 20
blocks = []

acc_points = []
count = 0

for t in lidar_ts:
    pts = np.load(lidar_dict[t])
    mask = np.any(pts != 0, axis=1)
    pts = pts[mask][:, :3]   # xyz only

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

# ======================================================================
# GLOBAL AXIS LIMITS — across all blocks and GT
# ======================================================================

all_xyz = []

# From blocks
for _, pts in blocks:
    xyz = pts[:, 1:4]
    if xyz.size:
        all_xyz.append(xyz)

# From GT
if gt_all_xyz.size:
    all_xyz.append(gt_all_xyz)

if len(all_xyz):
    all_xyz = np.vstack(all_xyz)
    Xmin, Xmax = np.min(all_xyz[:,0]), np.max(all_xyz[:,0])
    Ymin, Ymax = np.min(all_xyz[:,1]), np.max(all_xyz[:,1])
    Zmin, Zmax = np.min(all_xyz[:,2]), np.max(all_xyz[:,2])
else:
    Xmin = Ymin = Zmin = -10
    Xmax = Ymax = Zmax =  10

pad = 1.0
Xmin -= pad; Xmax += pad
Ymin -= pad; Ymax += pad
Zmin -= pad; Zmax += pad

# ======================================================================
# SETUP FIGURE (4 PANELS)
# ======================================================================

fig = plt.figure(figsize=(24, 6))

ax1 = fig.add_subplot(141, projection="3d")  # LiDAR360 + GT
ax2 = fig.add_subplot(142, projection="3d")  # Cluster-only
ax3 = fig.add_subplot(143, projection="3d")  # GT trajectory
ax4 = fig.add_subplot(144, projection="3d")  # NEW: Centroid vs GT trajectory

cluster_colors = [
    'red','green','blue','purple','orange','cyan','magenta','yellow'
]

# History storage
centroid_history = []
gt_history = []

# ======================================================================
# ANIMATION WRITER
# ======================================================================

out_video = os.path.join(OUT_DIR, f"{SEQ}_4panel_mid360_gt_centroid.mp4")
writer = FFMpegWriter(fps=1)
print("Saving MP4 to:", out_video)

# ======================================================================
# CSV OUTPUT
# ======================================================================

csv_path = os.path.join(OUT_DIR, f"{SEQ}_4panel_block_stats.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "block_idx", "t_start",
    "num_points", "num_clusters", "num_noise",
    "gt_cluster_id", "gt_error"
])

# ======================================================================
# ANIMATION LOOP
# ======================================================================

with writer.saving(fig, out_video, 200):

    for block_idx, (t_start, pts) in enumerate(blocks):

        # -------------------------------------------------------------
        # Reset axes
        # -------------------------------------------------------------
        for ax in (ax1, ax2, ax3, ax4):
            ax.cla()
            ax.set_xlim(Xmin, Xmax)
            ax.set_ylim(Ymin, Ymax)
            ax.set_zlim(Zmin, Zmax)

        # -------------------------------------------------------------
        # PANEL 1: Full LiDAR360 block + DBSCAN + GT
        # -------------------------------------------------------------
        xyz = pts[:,1:4]
        db = DBSCAN(eps=1, min_samples=10).fit(xyz)
        labels = db.labels_

        cluster_ids = sorted(set(labels))
        real_clusters = [c for c in cluster_ids if c != -1]
        num_noise = int(np.sum(labels == -1))

        # GT for this block
        gt_xyz = get_gt_for_timestamp(t_start)

        # Find cluster closest to GT
        gt_cluster_id = None
        min_dist = 1e9
        centroid = None

        for cid in real_clusters:
            cpts = xyz[labels == cid]
            c_mean = np.mean(cpts, axis=0)
            d = np.linalg.norm(c_mean - gt_xyz)
            if d < min_dist:
                min_dist = d
                centroid = c_mean
                gt_cluster_id = cid

        # PANELS 1–3 (unchanged from your script)
        # -------------- PANEL 1 plotting --------------
        plotted_any = False
        for cid in cluster_ids:
            cpts = xyz[labels == cid]
            if cpts.size == 0:
                continue
            plotted_any = True
            if cid == -1:
                ax1.scatter(cpts[:,0],cpts[:,1],cpts[:,2],s=3,c="lightgray",alpha=0.3)
            else:
                col = cluster_colors[cid % len(cluster_colors)]
                ax1.scatter(cpts[:,0],cpts[:,1],cpts[:,2],s=3,c=col,alpha=0.7)

        ax1.scatter(gt_xyz[0],gt_xyz[1],gt_xyz[2],s=120,c="red",marker="X")
        if centroid is not None:
            ax1.scatter(centroid[0],centroid[1],centroid[2],
                        s=200,c="yellow",marker="o",edgecolors="black")
        ax1.set_title("LiDAR360 Block (20 frames)")

        # -------------- PANEL 2 cluster-only ----------
        for cid in cluster_ids:
            cpts = xyz[labels == cid]
            if cpts.size == 0: continue
            if cid == -1:
                ax2.scatter(cpts[:,0],cpts[:,1],cpts[:,2],s=3,c="lightgray",alpha=0.3)
            else:
                col = cluster_colors[cid % len(cluster_colors)]
                ax2.scatter(cpts[:,0],cpts[:,1],cpts[:,2],s=3,c=col,alpha=0.7)
        ax2.set_title("Cluster-Only View")

        # -------------- PANEL 3 GT trajectory ----------
        if gt_all_xyz.size:
            hist_idx = np.where(gt_all_ts <= t_start)[0]
            if hist_idx.size:
                hist_xyz = gt_all_xyz[hist_idx]
                ax3.plot(hist_xyz[:,0],hist_xyz[:,1],hist_xyz[:,2],
                         linewidth=2,alpha=0.7,color="green")

        ax3.scatter(gt_xyz[0],gt_xyz[1],gt_xyz[2],
                    s=120,c="red",marker="X")
        ax3.set_title("Ground Truth Trajectory")

        # -------------------------------------------------------------
        # PANEL 4: Centroid vs Ground Truth trajectory (NEW)
        # -------------------------------------------------------------
        if centroid is not None:
            centroid_history.append(centroid)
            gt_history.append(gt_xyz)

        # Draw historical centroid positions
        if len(centroid_history) > 1:
            ch = np.vstack(centroid_history)
            ax4.plot(ch[:,0],ch[:,1],ch[:,2],
                     color="yellow", linewidth=2, label="Centroid Path")

        # Draw historical GT positions
        if len(gt_history) > 1:
            gh = np.vstack(gt_history)
            ax4.plot(gh[:,0],gh[:,1],gh[:,2],
                     color="green", linewidth=2, label="GT Path")

        # Current centroid and GT points
        if centroid is not None:
            ax4.scatter(centroid[0],centroid[1],centroid[2],
                        s=120,c="yellow",marker="o",edgecolors="black")
            ax4.scatter(gt_xyz[0],gt_xyz[1],gt_xyz[2],
                        s=120,c="red",marker="X")

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
            f"L360 pts: {xyz.shape[0]}\n"
            f"Clusters: {len(real_clusters)}\n"
            f"Noise pts: {num_noise}\n"
            f"GT cluster: {gt_cluster_id}\n"
            f"GT error: {min_dist:.3f} m"
        )
        ax1.text2D(0.02,0.98,overlay,transform=ax1.transAxes,
                   fontsize=9,verticalalignment="top")

        csv_writer.writerow([
            block_idx+1, t_start,
            int(xyz.shape[0]),
            len(real_clusters),
            num_noise,
            gt_cluster_id if gt_cluster_id is not None else -1,
            float(min_dist)
        ])

        writer.grab_frame()

csv_file.close()
plt.close(fig)

print("\n✓ 4-panel LiDAR360 + GT + Centroid animation saved.")
print("MP4:", out_video)
print("CSV:", csv_path)
