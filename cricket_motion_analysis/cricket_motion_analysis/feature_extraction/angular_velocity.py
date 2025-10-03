# Compute velocity / acceleration
import numpy as np

def angular_velocity(series, fps=30):
    """
    Angular velocity (deg/s).
    """
    series = np.asarray(series, dtype=float)
    vel = np.gradient(series) * fps
    return vel, {
        "vel_max": np.nanmax(vel),
        "vel_mean": np.nanmean(np.abs(vel))
    }

def angular_acceleration(velocity, fps=30):
    """
    Angular acceleration (deg/s²).
    """
    velocity = np.asarray(velocity, dtype=float)
    acc = np.gradient(velocity) * fps
    return acc, {
        "acc_max": np.nanmax(acc),
        "acc_mean": np.nanmean(np.abs(acc))
    }


def angular__velocity(angles, fps=30):
    """
    Compute frame-to-frame angular velocity
    angles: np.array of joint angles
    fps: frames per second of video
    """
    angles = np.array(angles)
    vel = np.gradient(angles) * fps
    return vel