"""
scripts/preprocess_data.py
--------------------------
One-time preprocessing: load raw jellyfish .mat and stickleback .h5 files,
run the full pipeline (area-change → bin-average → sliding windows → z-score),
and save compact .npz arrays to datasets/jellyfish_stickleback/.

Run from the repo root:
    python scripts/preprocess_data.py

--------------------------------------------------------------------------------
Step 1 — Constants & experiment metadata
--------------------------------------------------------------------------------
Two sampling rates and species-specific window lengths:

  Species      Native FPS   Window   Stride   Recording length
  Jellyfish    30 Hz        60 s     30 s     3600 s (1 hour)
  Stickleback  121 Hz       5 s      2 s      ~120 s

Jellyfish recordings last a full hour so long windows capture slow behavioral
drift during temperature ramps. Stickleback recordings are only ~120 s, so a
5-second window is used to generate enough training samples.

Both species downsample to 1 FPS (bin-average after differentiating at native
rate) before windowing, so the encoder always sees one sample per second.

The time axis for each jellyfish recording is aligned to the LED-on frame so
that t = 0 s is the moment the temperature ramp begins.

--------------------------------------------------------------------------------
Step 2 — atlas library
--------------------------------------------------------------------------------
Helper functions used from src/atlas/:

  atlas.io
    load_temp_csv        — thermocouple CSV loader
    load_sleap           — SLEAP .h5 loader → centroid speed at 1 FPS

  atlas.metrics
    extract_area_change  — |d(area)/dt| bin-averaged to 1 FPS

--------------------------------------------------------------------------------
Step 3 — Jellyfish windows
--------------------------------------------------------------------------------
For each of the four temperature-ramp experiments:

  1. Load the .mat file and compute |d(area)/dt| at 1 FPS via extract_area_change.
  2. Load the thermocouple CSV and interpolate measured temperature onto the
     1-FPS time grid. The time axis is aligned so t = 0 s is the LED-on frame.
  3. Slide a 60-second window (stride 30 s) across the recording.
  4. Label each window with the median measured temperature during that window.

Windows with >30% NaN frames are discarded. All jellyfish across all experiments
are pooled.

--------------------------------------------------------------------------------
Step 4 — Stickleback windows
--------------------------------------------------------------------------------
For each SLEAP .h5 file (four files × two temperatures = eight files total):

  1. Compute per-track centroid speed (px/ms) at 1 FPS via load_sleap.
  2. Slide a 5-second window (stride 2 s) across each track.
  3. Label each window with the temperature parsed from the folder name
     (17 °C or 22.5 °C).

A 5-second window at 1 FPS = 5 time steps fed to the encoder. AdaptiveAvgPool1d
in the encoder compresses any window length to the same 4-step summary, so both
species map to the same latent dimensionality regardless of window length.

Fish are free-swimming — centroid speed reflects actual swimming locomotion,
comparable to jellyfish |d(area)/dt| as a thermal activity metric.

--------------------------------------------------------------------------------
Step 5 — Per-species z-scoring
--------------------------------------------------------------------------------
Speed is z-scored separately per species. Jellyfish and fish have incomparable
raw units (px²/ms vs px/ms); a joint z-score would be dominated by whichever
species has larger variance. Per-species z-scoring places both on the same ±1–2
scale so the shared latent axis captures relative activity (high vs low for each
animal) rather than absolute magnitude.

Temperature labels are z-scored per species for the same reason: the jellyfish
ramp spans ~10–38 °C while fish labels are fixed at 17 °C and 22.5 °C.

After z-scoring, each window is reshaped to (N, T, 1) — T=60 for jellyfish,
T=5 for fish — and remaining NaNs are filled with 0 (the mean after z-scoring).
Means and standard deviations are saved so temperature predictions can be
converted back to °C in the notebook.

Output files (datasets/jellyfish_stickleback/):
    jellyfish.npz   — jelly_z, jelly_temps, jelly_temps_z, jelly_labels,
                      j_speed_mean, j_speed_std, j_temp_mean, j_temp_std
    fish.npz        — fish_z, fish_temps, fish_temps_z,
                      f_speed_mean, f_speed_std, f_temp_mean, f_temp_std

These files are tracked by git (small: ~1–2 MB total) so collaborators can
reproduce training without the raw data files (412 MB of .mat/.h5).
"""

import os, glob, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from atlas.io import load_temp_csv, load_sleap
from atlas.metrics import extract_area_change

# ── Constants ──────────────────────────────────────────────────────────────────
HULL_FPS     = 30
SLEAP_FPS    = 121
JELLY_WINDOW = 60
JELLY_STRIDE = 30
FISH_WINDOW  = 5
FISH_STRIDE  = 2

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAT_DIR      = os.path.join(ROOT, "data/Hull Objects (System20to40C_Crash_ System20to0C_Crash_ System20to36C_subCrash_ System20to50C_cook))")
TEMP_LOG_DIR = os.path.join(MAT_DIR, "Temperature Logs")
SLEAP_DIR    = os.path.join(ROOT, "data/stickleback_temperature_feb26")
OUT_DIR      = os.path.join(ROOT, "datasets/jellyfish_stickleback")

EXPERIMENTS = [
    {"label": "20→36°C sub-threshold crash (heating)",
     "mat_file": "011726_1150am_30FPS_System20to36C-01172026115024-0000_tracked.mat",
     "temp_csv": "2026-01-17_09_cl100.csv",
     "led_on_frame": 182, "tracking_start_frame": 0,   "cooling": False},
    {"label": "20→40°C crash (heating)",
     "mat_file": "012026_12PM_System20toSystem40C_30FPS-01202026120055-0000_tracked.mat",
     "temp_csv": "2026-01-20_07_cl100.csv",
     "led_on_frame": 122, "tracking_start_frame": 150, "cooling": False},
    {"label": "20→0°C crash (cooling)",
     "mat_file": "013026_1213pm_System20toSystem0_30FPS-01302026121402-0000_tracked.mat",
     "temp_csv": "2026-01-30_05_cl100.csv",
     "led_on_frame": 153, "tracking_start_frame": 182, "cooling": True},
    {"label": "20→50°C cook (heating)",
     "mat_file": "013026_220pm_System20toSystem50_30FPS-01302026142141-0000_tracked.mat",
     "temp_csv": "2026-01-30_06_cl100.csv",
     "led_on_frame": 161, "tracking_start_frame": 0,   "cooling": False},
]

# ── Jellyfish ──────────────────────────────────────────────────────────────────
print("Processing jellyfish ...")
jelly_windows, jelly_temps, jelly_labels = [], [], []

for exp in EXPERIMENTS:
    mat_path  = os.path.join(MAT_DIR,      exp['mat_file'])
    temp_path = os.path.join(TEMP_LOG_DIR, exp['temp_csv'])
    print(f"  {exp['label']}")

    area_speed          = extract_area_change(mat_path)
    t_csv, meas_temp, _ = load_temp_csv(temp_path)

    A = exp['led_on_frame']
    C = exp['tracking_start_frame']

    for j_idx, speed_1fps in area_speed.items():
        n_bins    = len(speed_1fps)
        bin_times = np.arange(n_bins) + (C - A) / HULL_FPS

        for w_start in range(0, n_bins - JELLY_WINDOW + 1, JELLY_STRIDE):
            win = speed_1fps[w_start : w_start + JELLY_WINDOW]
            if np.isnan(win).mean() > 0.3:
                continue
            t_mid      = bin_times[w_start : w_start + JELLY_WINDOW]
            temp_label = float(np.nanmedian(np.interp(t_mid, t_csv, meas_temp)))
            jelly_windows.append(win)
            jelly_temps.append(temp_label)
            jelly_labels.append(exp['label'])

jelly_windows = np.array(jelly_windows)
jelly_temps   = np.array(jelly_temps)
jelly_labels  = np.array(jelly_labels)

j_speed_mean = np.nanmean(jelly_windows);  j_speed_std = np.nanstd(jelly_windows)
jelly_z = np.nan_to_num(
    ((jelly_windows - j_speed_mean) / j_speed_std)[:, :, np.newaxis], nan=0.0
)

valid_j     = ~np.isnan(jelly_temps)
j_temp_mean = np.mean(jelly_temps[valid_j]);  j_temp_std = np.std(jelly_temps[valid_j])
jelly_temps_z = np.where(valid_j, (jelly_temps - j_temp_mean) / j_temp_std, np.nan)

print(f"  → {len(jelly_z)} windows, temp {jelly_temps.min():.1f}–{jelly_temps.max():.1f} °C")

# ── Stickleback ────────────────────────────────────────────────────────────────
print("Processing stickleback ...")
fish_windows, fish_temps = [], []

for temp_dir in sorted(os.listdir(SLEAP_DIR)):
    full_dir = os.path.join(SLEAP_DIR, temp_dir)
    if not os.path.isdir(full_dir):
        continue
    h5_files = sorted(glob.glob(os.path.join(full_dir, "*.h5")))
    if not h5_files:
        continue
    try:
        temp_val = float(temp_dir.split('C')[0].strip())
    except ValueError:
        continue
    print(f"  {temp_dir}  ({len(h5_files)} files)")

    for h5_path in h5_files:
        track_speeds = load_sleap(h5_path)
        for t_idx, speed_1fps in track_speeds.items():
            n_bins = len(speed_1fps)
            for w_start in range(0, n_bins - FISH_WINDOW + 1, FISH_STRIDE):
                win = speed_1fps[w_start : w_start + FISH_WINDOW]
                if np.isnan(win).mean() > 0.3:
                    continue
                fish_windows.append(win)
                fish_temps.append(temp_val)

fish_windows = np.array(fish_windows)
fish_temps   = np.array(fish_temps)

f_speed_mean = np.nanmean(fish_windows);  f_speed_std = np.nanstd(fish_windows)
fish_z = np.nan_to_num(
    ((fish_windows - f_speed_mean) / f_speed_std)[:, :, np.newaxis], nan=0.0
)

f_temp_mean = np.mean(fish_temps);  f_temp_std = np.std(fish_temps)
fish_temps_z = (fish_temps - f_temp_mean) / f_temp_std

print(f"  → {len(fish_z)} windows, temps {np.unique(fish_temps)}")

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

np.savez_compressed(
    os.path.join(OUT_DIR, "jellyfish.npz"),
    jelly_z=jelly_z,
    jelly_temps=jelly_temps,
    jelly_temps_z=jelly_temps_z,
    jelly_labels=jelly_labels,
    j_speed_mean=j_speed_mean, j_speed_std=j_speed_std,
    j_temp_mean=j_temp_mean,   j_temp_std=j_temp_std,
)

np.savez_compressed(
    os.path.join(OUT_DIR, "fish.npz"),
    fish_z=fish_z,
    fish_temps=fish_temps,
    fish_temps_z=fish_temps_z,
    f_speed_mean=f_speed_mean, f_speed_std=f_speed_std,
    f_temp_mean=f_temp_mean,   f_temp_std=f_temp_std,
)

jf = os.path.getsize(os.path.join(OUT_DIR, "jellyfish.npz")) / 1e6
ff = os.path.getsize(os.path.join(OUT_DIR, "fish.npz")) / 1e6
print(f"\nSaved to {OUT_DIR}/")
print(f"  jellyfish.npz  {jf:.2f} MB")
print(f"  fish.npz       {ff:.2f} MB")
