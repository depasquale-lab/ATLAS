"""
atlas.io
--------
File I/O helpers for jellyfish thermocouple logs and SLEAP pose files.
"""

import numpy as np
import h5py

_DEFAULT_BODY_NODES = ["body_1", "body_2", "body_3"]


def load_temp_csv(csv_path):
    """Load a thermocouple CSV file.

    Expected columns (with header row): time_s, measured_temp_C, setpoint_temp_C

    Returns
    -------
    t : np.ndarray  — time in seconds
    meas : np.ndarray  — measured temperature (°C)
    setpt : np.ndarray  — setpoint temperature (°C)
    """
    t, meas, setpt = [], [], []
    with open(csv_path) as fh:
        next(fh)  # skip header
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) == 3:
                t.append(float(parts[0]))
                meas.append(float(parts[1]))
                setpt.append(float(parts[2]))
    return np.array(t), np.array(meas), np.array(setpt)


def load_sleap(h5_path, body_nodes=None, sleap_fps=121):
    """Load a SLEAP .h5 file and return per-track centroid speed at 1 FPS.

    Pipeline
    --------
    1. Average *body_nodes* positions to compute a centroid at *sleap_fps*.
    2. Speed (px/ms) = ||Δcentroid|| × sleap_fps / 1000 at native FPS.
    3. Prepend NaN (align length to n_frames), then bin-average
       *sleap_fps* frames → 1 second.

    Parameters
    ----------
    h5_path : str
    body_nodes : list of str, optional
        Node names to average for the centroid. Defaults to
        ['body_1', 'body_2', 'body_3'].
    sleap_fps : int
        Native recording frame rate.

    Returns
    -------
    dict : {track_idx: np.ndarray of shape (n_seconds,)}
        Centroid speed in px/ms, bin-averaged to 1 FPS.
    """
    if body_nodes is None:
        body_nodes = _DEFAULT_BODY_NODES

    with h5py.File(h5_path, "r") as f:
        raw = f["tracks"][:]
        node_names = [n.decode() for n in f["node_names"][:]]
        track_names = [n.decode() for n in f["track_names"][:]]
        occupancy = f["track_occupancy"][:].astype(bool)

    tracks = raw.transpose(0, 3, 2, 1)  # → (track, frame, node, xy)
    for t in range(tracks.shape[0]):
        tracks[t, ~occupancy[:, t], :, :] = np.nan

    body_idx = [node_names.index(n) for n in body_nodes if n in node_names]

    result = {}
    for t_i in range(len(track_names)):
        body_xy = tracks[t_i][:, body_idx, :]
        centroid = np.nanmean(body_xy, axis=1)
        dx = np.diff(centroid[:, 0])
        dy = np.diff(centroid[:, 1])
        speed = np.sqrt(dx**2 + dy**2) * sleap_fps / 1000  # px/ms
        speed = np.concatenate([[np.nan], speed])
        n_bins = len(speed) // sleap_fps
        binned = np.array(
            [np.nanmean(speed[b * sleap_fps : (b + 1) * sleap_fps]) for b in range(n_bins)]
        )
        result[t_i] = binned

    return result


def load_sleap_tracks(h5_path):
    """Load a SLEAP .h5 file and return raw tracks at native FPS.

    Lower-level than :func:`load_sleap` — returns the full pose array rather
    than a binned speed summary. Useful for visualisation and custom metrics.

    Parameters
    ----------
    h5_path : str

    Returns
    -------
    tracks : np.ndarray of shape (n_tracks, n_frames, n_nodes, 2)
        x, y per node per frame. Unoccupied frames are NaN.
    node_names : list of str
    track_names : list of str
    occupancy : np.ndarray of shape (n_frames, n_tracks), bool
    """
    with h5py.File(h5_path, "r") as f:
        raw         = f["tracks"][:]
        node_names  = [n.decode() for n in f["node_names"][:]]
        track_names = [n.decode() for n in f["track_names"][:]]
        occupancy   = f["track_occupancy"][:].astype(bool)

    tracks = raw.transpose(0, 3, 2, 1)  # → (track, frame, node, xy)
    for t in range(tracks.shape[0]):
        tracks[t, ~occupancy[:, t], :, :] = np.nan

    return tracks, node_names, track_names, occupancy


def fish_centroid_speed(h5_path, body_nodes=None, sleap_fps=121):
    """Load a SLEAP .h5 file and return per-track centroid speed at native FPS.

    Unlike :func:`load_sleap`, this returns the full-resolution speed array
    (one value per frame) rather than bin-averaged seconds. Useful for
    fold-change comparisons and visualisation.

    Parameters
    ----------
    h5_path : str
    body_nodes : list of str, optional
        Node names to average for the centroid. Defaults to
        ['body_1', 'body_2', 'body_3'].
    sleap_fps : int
        Native recording frame rate (used for px/ms conversion).

    Returns
    -------
    list of np.ndarray
        One array per track, shape (n_frames - 1,), speed in px/ms.
        NaN where the track is unoccupied.
    """
    if body_nodes is None:
        body_nodes = _DEFAULT_BODY_NODES

    tracks, node_names, track_names, _ = load_sleap_tracks(h5_path)
    body_idx = [node_names.index(n) for n in body_nodes if n in node_names]

    speeds = []
    for t_i in range(len(track_names)):
        body_xy  = tracks[t_i][:, body_idx, :]
        centroid = np.nanmean(body_xy, axis=1)
        dx = np.diff(centroid[:, 0])
        dy = np.diff(centroid[:, 1])
        speed = np.sqrt(dx**2 + dy**2) * sleap_fps / 1000  # px/ms
        speeds.append(speed)

    return speeds
