# Limb symmetry and extremity speed
import numpy as np

def limb_symmetry(left_series, right_series):
    """
    Symmetry = mean absolute difference between left & right joint angles.
    """
    left = np.asarray(left_series, dtype=float)
    right = np.asarray(right_series, dtype=float)
    return np.nanmean(np.abs(left - right))

def extremity_speed(coords, fps=30):
    """
    Compute extremity speed (e.g., hand, bat, ball).
    coords: list of (x,y) per frame.
    Returns max & mean speed in pixels/s (or normalized units).
    """
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] < 2:
        return {"speed_max": np.nan, "speed_mean": np.nan}

    diffs = np.diff(coords, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    speeds = dists * fps
    return {
        "speed_max": np.nanmax(speeds),
        "speed_mean": np.nanmean(speeds)
    }

def calculate_symmetry_score(movement_data: dict) -> dict:
    """Calculate symmetry score comparing left/right joint angles from the movement data.

    For each joint that starts with 'left_', it looks for the corresponding 'right_' joint,
    and computes a symmetry score as 1 - (normalized mean difference). If paired joints are
    not found, a default symmetry score of 1.0 is returned for that joint.
    """
    scores = {}
    for joint in movement_data:
        if joint.startswith('left_'):
            counterpart = 'right_' + joint[5:]
            if counterpart in movement_data:
                left = np.array(movement_data[joint])
                right = np.array(movement_data[counterpart])
                avg = (left + right) / 2.0
                diff = np.abs(left - right)
                epsilon = 1e-6
                symmetry = 1 - np.mean(diff / (avg + epsilon))
                scores[joint[5:]] = symmetry
    if not scores:
        for joint, angles in movement_data.items():
            scores[joint] = 1.0
    return scores
