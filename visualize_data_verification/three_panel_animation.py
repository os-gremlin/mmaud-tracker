import os
import shutil
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter

plt.ioff()

# ======================================================================
# USER CONFIG
# ======================================================================

# Root folder that contains seq0001, seq0002, ...
ROOT_SEQ_DIR = r"C:/swarm/MMOAD/Calibration Data/train"

# Where to save timestamped result folders
RESULT_ROOT = r"C:/swarm/MMOAD/Calibration Data/results"

# Which sequence to visualize
SEQ = "seq0001"   # change to "seq0002", etc.

# Animation / render settings
FPS        = 30       # frames per second for output
EVERY      = 1        # step between frames (1 = use every frame)
POINT_SIZE = 2
PAD_FRAC   = 0.05
DPI        = 100

# GT anchoring behavior
# True  -> subtract first GT point so trajectory starts at (0,0,0)
# False -> use absolute GT coordinates
ANCHOR_GT_TO_ZERO = True

# If False, we only use timestamps common to all three.
# If True, we form a union of timestamps and use nearest neighbors per stream.
USE_NEAREST_MATCH = True

# ======================================================================
# PATH PREP
# ======================================================================

SEQ_DIR      = os.path.join(ROOT_SEQ_DIR, SEQ)
LIDAR360_DIR = os.path.join(SEQ_DIR, "lidar_360")
AVIA_DIR     = os.path.join(SEQ_DIR, "livox_avia")
GT_DIR       = os.path.join(SEQ_DIR, "ground_truth")

if not os.path.isdir(SEQ_DIR):
    raise FileNotFoundError(f"Sequence folder not found: {SEQ_DIR}")
if not os.path.isdir(LIDAR360_DIR):
    raise FileNotFoundError(f"Missing lidar_360 folder: {LIDAR360_DIR}")
if not os.path.isdir(AVIA_DIR):
    raise FileNotFoundError(f"Missing livox_avia folder: {AVIA_DIR}")
if not os.path.isdir(GT_DIR):
    raise FileNotFoundError(f"Missing ground_truth folder: {GT_DIR}")

use_ffmpeg = shutil.which("ffmpeg") is not None
timestamp_run = datetime.now().strftime("%Y%m%d_%H%M%S")

EXPORT_DIR = os.path.join(RESULT_ROOT, timestamp_run)
os.makedirs(EXPORT_DIR, exist_ok=True)

EXT = "mp4" if use_ffmpeg else "gif"

OUT_TRIPLE = os.path.join(
    EXPORT_DIR,
    f"{SEQ}_triple_view_lidar360_avia_gt.{EXT}"
)
OUT_TRIPLE_REFLECT = os.path.join(
    EXPORT_DIR,
    f"{SEQ}_triple_view_lidar360_avia_gt_reflection.{EXT}"
)

print(f"Export directory: {EXPORT_DIR}")

# ======================================================================
# HELPERS FOR LOADING AND TIMESTAMPS
# ======================================================================

def list_timestamp_files(folder):
    """Return (timestamps_float, path_dict) for .npy files in folder."""
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    ts_float = []
    path_map = {}
    for f in files:
        stem = os.path.splitext(f)[0]
        try:
            t = float(stem)
        except ValueError:
            continue
        full = os.path.join(folder, f)
        ts_float.append(t)
        path_map[t] = full
    ts_float = np.array(sorted(ts_float))
    return ts_float, path_map

def nearest_idx(target, arr):
    """Return index of nearest value in arr to target. arr must be 1D np.array."""
    if arr.size == 0:
        return None
    idx = np.abs(arr - target).argmin()
    return int(idx)

def load_xyz_and_intensity(path):
    """Load a .npy file that may be:
       - raw array of shape (N,3) or (N,4)
       - dict with key 'points'
    """
    data = np.load(path, allow_pickle=True)

    # Case 1: dict-style { "points" : array }
    if isinstance(data, dict):
        pts = data.get("points")
        if pts is None:
            raise ValueError(f"'points' key missing in dict from {path}")
        pts = np.asarray(pts)

    # Case 2: raw array
    else:
        pts = np.asarray(data)

    # Might be zeros-padded; shape must be Nx3 or Nx4
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Unexpected point shape from {path}: {pts.shape}")

    # Optional: filter out pure zeros
    # pts = pts[~np.all(pts == 0, axis=1)]

    xyz = pts[:, :3]
    I = pts[:, 3] if pts.shape[1] >= 4 else None
    return xyz, I

def maybe_get_xyz_at(lidar_ts_array, lidar_paths_dict, t):
    """Get xyz for nearest timestamp for given time t, or None if empty."""
    if lidar_ts_array.size == 0:
        return None
    idx = nearest_idx(t, lidar_ts_array)
    if idx is None:
        return None
    t_near = lidar_ts_array[idx]
    path = lidar_paths_dict[t_near]
    xyz, _ = load_xyz_and_intensity(path)
    return xyz

# ======================================================================
# LOAD TIMESTAMPS AND GROUND TRUTH
# ======================================================================

l360_ts, l360_paths = list_timestamp_files(LIDAR360_DIR)
avia_ts, avia_paths = list_timestamp_files(AVIA_DIR)
gt_ts,   gt_paths   = list_timestamp_files(GT_DIR)

if gt_ts.size == 0:
    raise RuntimeError("No GT .npy files found in ground_truth folder.")

# Load all GT positions into time-sorted arrays
gt_ts_sorted = np.sort(gt_ts)
gt_full = []
for t in gt_ts_sorted:
    g = np.load(gt_paths[t])
    g = np.asarray(g, dtype=float).reshape(3,)
    gt_full.append(g)
gt_full = np.vstack(gt_full)  # shape (N_gt, 3)

# Anchor or absolute GT
if ANCHOR_GT_TO_ZERO:
    gt_origin = gt_full[0].copy()
    gt_full_anchored = gt_full - gt_origin
    print("GT will be anchored to (0,0,0) at first frame.")
else:
    gt_origin = np.zeros(3)
    gt_full_anchored = gt_full.copy()
    print("GT will use absolute coordinates (no anchoring).")

def gt_for_time(t):
    """Return anchored GT for the timestamp t (via nearest GT timestamp)."""
    idx = nearest_idx(t, gt_ts_sorted)
    if idx is None:
        return None
    return gt_full_anchored[idx]

# Master timeline for animation
if USE_NEAREST_MATCH:
    all_ts = np.array(sorted(set(l360_ts.tolist()) |
                             set(avia_ts.tolist()) |
                             set(gt_ts_sorted.tolist())))
else:
    common = set(l360_ts.tolist()) & set(avia_ts.tolist()) & set(gt_ts_sorted.tolist())
    all_ts = np.array(sorted(common))

if all_ts.size == 0:
    raise RuntimeError("No timestamps found for animation.")

# ======================================================================
# COMPUTE GLOBAL BOUNDS FOR AXES
# ======================================================================

def compute_bounds():
    xyz_all = []
    step = max(1, all_ts.size // 200)  # sample up to ~200 frames

    for i in range(0, all_ts.size, step):
        t = all_ts[i]

        xyz_mid = maybe_get_xyz_at(l360_ts, l360_paths, t)
        if xyz_mid is not None and xyz_mid.size > 0:
            xyz_all.append(xyz_mid)

        xyz_avia = maybe_get_xyz_at(avia_ts, avia_paths, t)
        if xyz_avia is not None and xyz_avia.size > 0:
            xyz_all.append(xyz_avia)

    if not xyz_all:
        raise RuntimeError("No valid LiDAR points found to compute bounds.")

    xyz_all_cat = np.vstack(xyz_all)
    xs, ys, zs = xyz_all_cat[:, 0], xyz_all_cat[:, 1], xyz_all_cat[:, 2]

    xs_min, xs_max = np.percentile(xs, [1, 99])
    ys_min, ys_max = np.percentile(ys, [1, 99])
    zs_min, zs_max = np.percentile(zs, [1, 99])

    xr = xs_max - xs_min
    yr = ys_max - ys_min
    zr = zs_max - zs_min

    xs_min -= PAD_FRAC * xr
    xs_max += PAD_FRAC * xr
    ys_min -= PAD_FRAC * yr
    ys_max += PAD_FRAC * yr
    zs_min -= PAD_FRAC * zr
    zs_max += PAD_FRAC * zr

    return (xs_min, xs_max, ys_min, ys_max, zs_min, zs_max, xr, yr, zr)

(xs_min, xs_max,
 ys_min, ys_max,
 zs_min, zs_max,
 xr, yr, zr) = compute_bounds()

def setup_axes_3d(ax):
    ax.set_xlim(xs_min, xs_max)
    ax.set_ylim(ys_min, ys_max)
    ax.set_zlim(zs_min, zs_max)
    ax.set_box_aspect((xr, yr, zr))
    ax.view_init(20, -60)

# Precompute GT along the all_ts timeline for easy trajectory drawing
gt_timeline = []
for t in all_ts:
    g = gt_for_time(t)
    if g is None:
        g = np.array([np.nan, np.nan, np.nan])
    gt_timeline.append(g)
gt_timeline = np.vstack(gt_timeline)  # shape (len(all_ts), 3)

n_frames = len(all_ts)
print(f"Using {n_frames} frames based on timestamps.")

Writer = FFMpegWriter if use_ffmpeg else PillowWriter

# ======================================================================
# TRIPLE-VIEW ANIMATION (NORMAL)
# ======================================================================

fig_triple = plt.figure(figsize=(18, 6))

ax_l360 = fig_triple.add_subplot(131, projection="3d")
ax_avia = fig_triple.add_subplot(132, projection="3d")
ax_gt3d = fig_triple.add_subplot(133, projection="3d")

for axv in [ax_l360, ax_avia, ax_gt3d]:
    setup_axes_3d(axv)

ax_l360.set_title("LiDAR 360")
ax_avia.set_title("Livox Avia")
ax_gt3d.set_title("GT (anchored)" if ANCHOR_GT_TO_ZERO else "GT (absolute)")

(gt_dot_3v,) = ax_gt3d.plot([], [], [], "ro")
(gt_traj_3v,) = ax_gt3d.plot([], [], [], "r--")

w_triple = Writer(fps=FPS)

with w_triple.saving(fig_triple, OUT_TRIPLE, DPI):
    for frame_idx in range(0, n_frames, EVERY):
        t = all_ts[frame_idx]

        # Clear LiDAR scatters only
        for c in list(ax_l360.collections):
            c.remove()
        for c in list(ax_avia.collections):
            c.remove()

        # LIDAR360 PANEL
        xyz_mid = maybe_get_xyz_at(l360_ts, l360_paths, t)
        if xyz_mid is not None and xyz_mid.size > 0:
            ax_l360.scatter(
                xyz_mid[:, 0], xyz_mid[:, 1], xyz_mid[:, 2],
                s=POINT_SIZE, c="gray", alpha=0.7
            )

        # AVIA PANEL
        xyz_avia = maybe_get_xyz_at(avia_ts, avia_paths, t)
        if xyz_avia is not None and xyz_avia.size > 0:
            ax_avia.scatter(
                xyz_avia[:, 0], xyz_avia[:, 1], xyz_avia[:, 2],
                s=POINT_SIZE, c="cyan", alpha=0.5
            )

        # GT PANEL
        g = gt_timeline[frame_idx]
        if not np.isnan(g).any():
            gt_dot_3v.set_data([g[0]], [g[1]])
            gt_dot_3v.set_3d_properties([g[2]])

            traj = gt_timeline[:frame_idx + 1]
            traj = traj[~np.isnan(traj).any(axis=1)]
            if traj.size > 0:
                gt_traj_3v.set_data(traj[:, 0], traj[:, 1])
                gt_traj_3v.set_3d_properties(traj[:, 2])

        fig_triple.suptitle(
            f"{SEQ} Triple-View | Frame {frame_idx+1}/{n_frames}",
            fontsize=16
        )

        w_triple.grab_frame()

plt.close(fig_triple)

# ======================================================================
# TRIPLE-VIEW REFLECTION (MIRROR X)
# ======================================================================

fig_tr = plt.figure(figsize=(18, 6))

ax_l360_r = fig_tr.add_subplot(131, projection="3d")
ax_avia_r = fig_tr.add_subplot(132, projection="3d")
ax_gt3d_r = fig_tr.add_subplot(133, projection="3d")

for axv in [ax_l360_r, ax_avia_r, ax_gt3d_r]:
    setup_axes_3d(axv)

ax_l360_r.set_title("LiDAR 360 (reflected X)")
ax_avia_r.set_title("Livox Avia (reflected X)")
ax_gt3d_r.set_title("GT (reflected X)")

(gt_dot_r,) = ax_gt3d_r.plot([], [], [], "ro")
(gt_traj_r,) = ax_gt3d_r.plot([], [], [], "r--")

w_ref = Writer(fps=FPS)

with w_ref.saving(fig_tr, OUT_TRIPLE_REFLECT, DPI):
    for frame_idx in range(0, n_frames, EVERY):
        t = all_ts[frame_idx]

        # Clear LiDAR scatters only
        for c in list(ax_l360_r.collections):
            c.remove()
        for c in list(ax_avia_r.collections):
            c.remove()

        # LiDAR360 REFLECTED
        xyz_mid = maybe_get_xyz_at(l360_ts, l360_paths, t)
        if xyz_mid is not None and xyz_mid.size > 0:
            xyz_mid_ref = xyz_mid.copy()
            xyz_mid_ref[:, 0] *= -1
            ax_l360_r.scatter(
                xyz_mid_ref[:, 0], xyz_mid_ref[:, 1], xyz_mid_ref[:, 2],
                s=POINT_SIZE, c="gray", alpha=0.7
            )

        # AVIA REFLECTED
        xyz_avia = maybe_get_xyz_at(avia_ts, avia_paths, t)
        if xyz_avia is not None and xyz_avia.size > 0:
            xyz_avia_ref = xyz_avia.copy()
            xyz_avia_ref[:, 0] *= -1
            ax_avia_r.scatter(
                xyz_avia_ref[:, 0], xyz_avia_ref[:, 1], xyz_avia_ref[:, 2],
                s=POINT_SIZE, c="cyan", alpha=0.5
            )

        # GT REFLECTED
        g = gt_timeline[frame_idx]
        if not np.isnan(g).any():
            g_ref = g.copy()
            g_ref[0] *= -1

            gt_dot_r.set_data([g_ref[0]], [g_ref[1]])
            gt_dot_r.set_3d_properties([g_ref[2]])

            traj = gt_timeline[:frame_idx + 1].copy()
            traj = traj[~np.isnan(traj).any(axis=1)]
            if traj.size > 0:
                traj[:, 0] *= -1
                gt_traj_r.set_data(traj[:, 0], traj[:, 1])
                gt_traj_r.set_3d_properties(traj[:, 2])

        fig_tr.suptitle(
            f"{SEQ} Triple-View REFLECTION | Frame {frame_idx+1}/{n_frames}",
            fontsize=16
        )

        w_ref.grab_frame()

plt.close(fig_tr)

# ======================================================================
# SINGLE-VIEW ANIMATIONS (LiDAR360, AVIA, GT-only)
# ======================================================================

OUT_L360 = os.path.join(EXPORT_DIR, f"{SEQ}_lidar360_only.{EXT}")
OUT_AVIA = os.path.join(EXPORT_DIR, f"{SEQ}_avia_only.{EXT}")
OUT_GT_ONLY = os.path.join(EXPORT_DIR, f"{SEQ}_gt_only.{EXT}")

# -------- LIDAR 360 ONLY ----------------------------------------------
fig_l360 = plt.figure(figsize=(8, 6))
ax_l360_only = fig_l360.add_subplot(111, projection="3d")
setup_axes_3d(ax_l360_only)
ax_l360_only.set_title("LiDAR 360 Only")

w_l360 = Writer(fps=FPS)
with w_l360.saving(fig_l360, OUT_L360, DPI):
    for frame_idx in range(0, n_frames, EVERY):
        t = all_ts[frame_idx]

        for c in list(ax_l360_only.collections):
            c.remove()

        xyz_mid = maybe_get_xyz_at(l360_ts, l360_paths, t)
        if xyz_mid is not None and xyz_mid.size > 0:
            ax_l360_only.scatter(
                xyz_mid[:, 0], xyz_mid[:, 1], xyz_mid[:, 2],
                s=POINT_SIZE, c="gray", alpha=0.7
            )

        ax_l360_only.set_title(f"LiDAR 360 Only | Frame {frame_idx+1}/{n_frames}")
        w_l360.grab_frame()
plt.close(fig_l360)

# -------- AVIA ONLY ----------------------------------------------------
fig_avia_only = plt.figure(figsize=(8, 6))
ax_avia_only = fig_avia_only.add_subplot(111, projection="3d")
setup_axes_3d(ax_avia_only)
ax_avia_only.set_title("Livox Avia Only")

w_avia = Writer(fps=FPS)
with w_avia.saving(fig_avia_only, OUT_AVIA, DPI):
    for frame_idx in range(0, n_frames, EVERY):
        t = all_ts[frame_idx]

        for c in list(ax_avia_only.collections):
            c.remove()

        xyz_avia = maybe_get_xyz_at(avia_ts, avia_paths, t)
        if xyz_avia is not None and xyz_avia.size > 0:
            ax_avia_only.scatter(
                xyz_avia[:, 0], xyz_avia[:, 1], xyz_avia[:, 2],
                s=POINT_SIZE, c="cyan", alpha=0.5
            )

        ax_avia_only.set_title(f"Livox Avia Only | Frame {frame_idx+1}/{n_frames}")
        w_avia.grab_frame()
plt.close(fig_avia_only)

# -------- GT ONLY ------------------------------------------------------
fig_gt_only = plt.figure(figsize=(8, 6))
ax_gt_only = fig_gt_only.add_subplot(111, projection="3d")
setup_axes_3d(ax_gt_only)
ax_gt_only.set_title("GT Only")

(gt_dot_only,) = ax_gt_only.plot([], [], [], "ro")
(gt_traj_only,) = ax_gt_only.plot([], [], [], "r--")

w_gt = Writer(fps=FPS)
with w_gt.saving(fig_gt_only, OUT_GT_ONLY, DPI):
    for frame_idx in range(0, n_frames, EVERY):
        g = gt_timeline[frame_idx]

        gt_dot_only.set_data([g[0]], [g[1]])
        gt_dot_only.set_3d_properties([g[2]])

        traj = gt_timeline[:frame_idx+1]
        traj = traj[~np.isnan(traj).any(axis=1)]
        if traj.size > 0:
            gt_traj_only.set_data(traj[:, 0], traj[:, 1])
            gt_traj_only.set_3d_properties(traj[:, 2])

        ax_gt_only.set_title(f"GT Only | Frame {frame_idx+1}/{n_frames}")
        w_gt.grab_frame()
plt.close(fig_gt_only)

# ======================================================================
# STATIC PNG PLOTS
# ======================================================================

# Middle frame time
mid_idx = n_frames // 2
mid_t = all_ts[mid_idx]

xyz_mid_mid = maybe_get_xyz_at(l360_ts, l360_paths, mid_t)
xyz_avia_mid = maybe_get_xyz_at(avia_ts, avia_paths, mid_t)
g_mid = gt_timeline[mid_idx]

# 1) LiDAR360 + single GT point
fig_s1 = plt.figure(figsize=(8, 6))
ax_s1 = fig_s1.add_subplot(111, projection="3d")
setup_axes_3d(ax_s1)
if xyz_mid_mid is not None and xyz_mid_mid.size > 0:
    ax_s1.scatter(xyz_mid_mid[:, 0], xyz_mid_mid[:, 1], xyz_mid_mid[:, 2],
                  s=1, c="gray")
ax_s1.scatter(g_mid[0], g_mid[1], g_mid[2], c="red", s=40)
ax_s1.set_title("LiDAR 360 + GT (single frame)")
out1 = os.path.join(EXPORT_DIR, f"{SEQ}_lidar360_single_frame.png")
fig_s1.savefig(out1, dpi=150)
plt.close(fig_s1)

# 2) AVIA + single GT point
fig_s2 = plt.figure(figsize=(8, 6))
ax_s2 = fig_s2.add_subplot(111, projection="3d")
setup_axes_3d(ax_s2)
if xyz_avia_mid is not None and xyz_avia_mid.size > 0:
    ax_s2.scatter(xyz_avia_mid[:, 0], xyz_avia_mid[:, 1], xyz_avia_mid[:, 2],
                  s=1, c="cyan")
ax_s2.scatter(g_mid[0], g_mid[1], g_mid[2], c="red", s=40)
ax_s2.set_title("Livox Avia + GT (single frame)")
out2 = os.path.join(EXPORT_DIR, f"{SEQ}_avia_single_frame.png")
fig_s2.savefig(out2, dpi=150)
plt.close(fig_s2)

# 3) GT full trajectory
fig_s3 = plt.figure(figsize=(8, 6))
ax_s3 = fig_s3.add_subplot(111, projection="3d")
setup_axes_3d(ax_s3)
ax_s3.plot(gt_timeline[:, 0], gt_timeline[:, 1], gt_timeline[:, 2],
           "r-", lw=2)
ax_s3.set_title("GT Full Trajectory")
out3 = os.path.join(EXPORT_DIR, f"{SEQ}_gt_full_traj.png")
fig_s3.savefig(out3, dpi=150)
plt.close(fig_s3)

print("✅ Done.")
print("Triple-view:          ", OUT_TRIPLE)
print("Triple-view reflected:", OUT_TRIPLE_REFLECT)
print("LiDAR360 only:        ", OUT_L360)
print("Avia only:            ", OUT_AVIA)
print("GT only:              ", OUT_GT_ONLY)
print("Static PNGs:")
print("   ", out1)
print("   ", out2)
print("   ", out3)
print(f"All outputs exported to:\n{EXPORT_DIR}")
 