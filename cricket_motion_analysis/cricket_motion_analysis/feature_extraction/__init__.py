import pandas as pd
from .joint_features import joint_stats, range_of_motion
from .angular_velocity import angular_velocity, angular_acceleration
from .symmetry_speed import limb_symmetry, extremity_speed

def extract_features(df, fps=30):
    """
    Extract features from a preprocessed joint angle dataframe.
    df: DataFrame with 'frame' + joint angle columns.
    """
    features = {}

    for joint in df.columns:
        if joint == "frame" or joint == "phase":
            continue
        series = df[joint].values

        # Joint stats
        stats = joint_stats(series)
        features.update({f"{joint}_{k}": v for k, v in stats.items()})

        # ROM
        features[f"{joint}_ROM"] = range_of_motion(series)

        # Velocity & acceleration
        vel, vel_stats = angular_velocity(series, fps=fps)
        acc, acc_stats = angular_acceleration(vel, fps=fps)
        features.update({f"{joint}_{k}": v for k, v in vel_stats.items()})
        features.update({f"{joint}_{k}": v for k, v in acc_stats.items()})

    # Symmetry (example: elbows, knees if present)
    if "left_elbow" in df.columns and "right_elbow" in df.columns:
        features["elbow_symmetry"] = limb_symmetry(df["left_elbow"], df["right_elbow"])
    if "left_knee" in df.columns and "right_knee" in df.columns:
        features["knee_symmetry"] = limb_symmetry(df["left_knee"], df["right_knee"])

    return pd.Series(features)
