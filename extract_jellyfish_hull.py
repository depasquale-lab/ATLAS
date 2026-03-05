"""
Extract hull data from jellyfish .mat files and save as compact HDF5.

Each .mat file contains a 'hull' variable: shape (1, n_jellyfish), where each
element is a structured array of shape (1, n_frames) with fields:
  Centroid (2,), MajorAxisLength (1,), MinorAxisLength (1,),
  BoundingBox (4,), Orientation (1,), Extrema (8,2), ConvexHull (variable, 2)

Output HDF5 structure per file:
  /centroid          (n_jellyfish, n_frames, 2)       float32
  /major_axis        (n_jellyfish, n_frames)           float32
  /minor_axis        (n_jellyfish, n_frames)           float32
  /orientation       (n_jellyfish, n_frames)           float32
  /bounding_box      (n_jellyfish, n_frames, 4)        float32
  /extrema           (n_jellyfish, n_frames, 8, 2)     float32
  /convex_hull       (n_jellyfish, n_frames, max_v, 2) float32  (NaN-padded)
  /convex_hull_nv    (n_jellyfish, n_frames)           uint8    (vertex count)
"""

import os
import glob
import numpy as np
import scipy.io as sio
import h5py

MAT_DIR = "data/Hull Objects (System20to40C_Crash_ System20to0C_Crash_ System20to36C_subCrash_ System20to50C_cook))"
OUT_DIR = "data/jellyfish_hull_h5"
os.makedirs(OUT_DIR, exist_ok=True)

mat_files = sorted(glob.glob(os.path.join(MAT_DIR, "*.mat")))
print(f"Found {len(mat_files)} .mat files\n")

for mat_path in mat_files:
    basename = os.path.splitext(os.path.basename(mat_path))[0]
    out_path = os.path.join(OUT_DIR, basename + ".h5")

    print(f"Processing: {os.path.basename(mat_path)}")
    print(f"  Loading .mat ... ", end="", flush=True)
    mat = sio.loadmat(mat_path)
    hull = mat["hull"]  # (1, n_jellyfish)
    n_jelly = hull.shape[1]
    n_frames = hull[0, 0].shape[1]
    print(f"done  ({n_jelly} jellyfish, {n_frames} frames)")

    # First pass: find max convex hull vertex count across all jellyfish
    print(f"  Finding max hull vertices ... ", end="", flush=True)
    max_v = 0
    for j in range(n_jelly):
        jelly = hull[0, j]
        for f in range(0, n_frames, 50):  # sample every 50 frames
            v = jelly["ConvexHull"][0, f].shape[0]
            if v > max_v:
                max_v = v
    # Add a small buffer to be safe
    max_v += 2
    print(f"done  (max_v={max_v})")

    # Allocate output arrays
    centroid   = np.full((n_jelly, n_frames, 2),       np.nan, dtype=np.float32)
    major_axis = np.full((n_jelly, n_frames),           np.nan, dtype=np.float32)
    minor_axis = np.full((n_jelly, n_frames),           np.nan, dtype=np.float32)
    orientation= np.full((n_jelly, n_frames),           np.nan, dtype=np.float32)
    bbox       = np.full((n_jelly, n_frames, 4),        np.nan, dtype=np.float32)
    extrema    = np.full((n_jelly, n_frames, 8, 2),     np.nan, dtype=np.float32)
    ch         = np.full((n_jelly, n_frames, max_v, 2), np.nan, dtype=np.float32)
    ch_nv      = np.zeros((n_jelly, n_frames),                  dtype=np.uint8)

    # Extract
    for j in range(n_jelly):
        jelly = hull[0, j]
        print(f"  Extracting jellyfish {j+1}/{n_jelly} ... ", end="", flush=True)
        for f in range(n_frames):
            centroid[j, f]    = jelly["Centroid"][0, f].ravel()
            major_axis[j, f]  = jelly["MajorAxisLength"][0, f].ravel()[0]
            minor_axis[j, f]  = jelly["MinorAxisLength"][0, f].ravel()[0]
            orientation[j, f] = jelly["Orientation"][0, f].ravel()[0]
            bbox[j, f]        = jelly["BoundingBox"][0, f].ravel()
            extrema[j, f]     = jelly["Extrema"][0, f]
            verts = jelly["ConvexHull"][0, f]
            nv = verts.shape[0]
            ch[j, f, :nv] = verts
            ch_nv[j, f]   = nv
        print("done")

    # Save to HDF5
    print(f"  Saving to {out_path} ... ", end="", flush=True)
    with h5py.File(out_path, "w") as hf:
        opts = dict(compression="gzip", compression_opts=6)
        hf.create_dataset("centroid",       data=centroid,    **opts)
        hf.create_dataset("major_axis",     data=major_axis,  **opts)
        hf.create_dataset("minor_axis",     data=minor_axis,  **opts)
        hf.create_dataset("orientation",    data=orientation, **opts)
        hf.create_dataset("bounding_box",   data=bbox,        **opts)
        hf.create_dataset("extrema",        data=extrema,     **opts)
        hf.create_dataset("convex_hull",    data=ch,          **opts)
        hf.create_dataset("convex_hull_nv", data=ch_nv,       **opts)

    in_mb  = os.path.getsize(mat_path) / 1e6
    out_mb = os.path.getsize(out_path) / 1e6
    print(f"done  ({in_mb:.1f}MB -> {out_mb:.1f}MB, {100*out_mb/in_mb:.1f}% of original)\n")

print("All done.")
