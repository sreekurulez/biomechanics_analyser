import numpy as np

def normalize_body_size(coords, ref_length):
    """
    Normalize joint coordinates based on a reference body size.
    coords: list of (x,y) or (x,y,z)
    ref_length: scalar (e.g., distance between shoulders or hips)
    """
    if ref_length == 0:
        return coords
    return [(x/ref_length, y/ref_length) for (x,y) in coords]

def recenter_pose(coords, center_joint):
    """
    Recenter pose so that a chosen joint (e.g., mid-hip) is at (0,0).
    """
    cx, cy = center_joint
    return [(x-cx, y-cy) for (x,y) in coords]

def normalize_time(series, target_len=100):
    """
    Resample time-series data to a fixed number of steps.
    """
    x = np.linspace(0, 1, len(series))
    new_x = np.linspace(0, 1, target_len)
    return np.interp(new_x, x, series)

def normalize_reference_pattern(reference_data: dict, target_length: int) -> dict:
    """
    Normalize reference pattern data to match the target length of captured data.
    
    Args:
        reference_data: Dictionary of joint angles from reference pattern
        target_length: Target number of frames to normalize to
        
    Returns:
        Dictionary of normalized joint angles matching target length
    """
    normalized_data = {}
    for joint, angles in reference_data.items():
        if joint != 'frame':  # Skip frame column
            normalized_data[joint] = normalize_time(angles, target_length)
    return normalized_data
