"""
atlas.metrics
-------------
Hull-geometry and speed metrics for jellyfish and stickleback data.
"""

import numpy as np
import scipy.io
import pandas as pd


def rolling_mean(arr, window):
    """Centred rolling mean with pandas, handling NaNs at edges.

    Parameters
    ----------
    arr : array-like
    window : int  — number of samples in the rolling window

    Returns
    -------
    np.ndarray of same length as *arr*
    """
    return pd.Series(arr).rolling(window, center=True, min_periods=1).mean().to_numpy()


def hull_area_shoelace(poly):
    """Area of a polygon using the shoelace formula.

    Parameters
    ----------
    poly : np.ndarray of shape (n_vertices, 2)

    Returns
    -------
    float
    """
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def extract_hull_area(mat_path):
    """Load a jellyfish .mat file and return raw convex-hull area per frame.

    Parameters
    ----------
    mat_path : str

    Returns
    -------
    dict : {jellyfish_idx: np.ndarray of shape (n_frames,)}
        Area in px² at native FPS (30 Hz). NaN for frames with no valid hull.
    """
    data = scipy.io.loadmat(mat_path)
    hulls = data["hull"]
    n_jelly = hulls.shape[1]
    result = {}
    for j in range(n_jelly):
        h = hulls[0, j]
        polys = h["ConvexHull"][0]
        area = np.array(
            [hull_area_shoelace(poly) if poly.size >= 6 else np.nan for poly in polys]
        )
        result[j] = area
    return result


def extract_area_change(mat_path, hull_fps=30):
    """Load a jellyfish .mat file and return |d(area)/dt| bin-averaged to 1 FPS.

    Pipeline
    --------
    1. Compute convex-hull area per frame (shoelace) at *hull_fps*.
    2. Differentiate: |d(area)/dt| in px²/frame at native FPS.
    3. Prepend NaN (align to n_frames), then bin-average *hull_fps* frames → 1 s.

    Parameters
    ----------
    mat_path : str
    hull_fps : int
        Native hull tracking frame rate.

    Returns
    -------
    dict : {jellyfish_idx: np.ndarray of shape (n_seconds,)}
        |d(area)/dt| in px²/frame, bin-averaged to 1 FPS.
    """
    data = scipy.io.loadmat(mat_path)
    hulls = data["hull"]
    n_jelly = hulls.shape[1]
    result = {}
    for j in range(n_jelly):
        h = hulls[0, j]
        polys = h["ConvexHull"][0]
        area = np.array(
            [hull_area_shoelace(poly) if poly.size >= 6 else np.nan for poly in polys]
        )
        darea = np.abs(np.diff(area))
        darea = np.concatenate([[np.nan], darea])  # length = n_frames
        n_bins = len(darea) // hull_fps
        binned = np.array(
            [np.nanmean(darea[b * hull_fps : (b + 1) * hull_fps]) for b in range(n_bins)]
        )
        result[j] = binned
    return result
