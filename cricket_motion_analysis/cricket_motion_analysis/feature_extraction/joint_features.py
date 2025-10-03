# Max, min, avg angles, ROM
import numpy as np
import pandas as pd

def joint_stats(series):
    """
    Compute basic stats: max, min, mean.
    series: list/np.array of joint angles.
    """
    # Convert to numeric, set errors to NaN
    series = pd.to_numeric(series, errors='coerce')
    series = np.asarray(series, dtype=float)
    
    # Handle empty or all-NaN series
    if len(series) == 0 or np.all(np.isnan(series)):
        return {
            "max": 0.0,
            "min": 0.0,
            "mean": 0.0
        }
    
    return {
        "max": np.nanmax(series),
        "min": np.nanmin(series),
        "mean": np.nanmean(series)
    }

def range_of_motion(series):
    """
    Compute range of motion (ROM = max - min).
    """
    # Convert to numeric, set errors to NaN
    series = pd.to_numeric(series, errors='coerce')
    series = np.asarray(series, dtype=float)
    
    # Handle empty or all-NaN series
    if len(series) == 0 or np.all(np.isnan(series)):
        return 0.0
        
    return np.nanmax(series) - np.nanmin(series)

def wrist_flex_analysis(wrist_angle_series, is_spin=False):
    """
    Analyze wrist flexion patterns, with special consideration for spin bowling.
    Args:
        wrist_angle_series: Series of wrist angles over time
        is_spin: Boolean indicating if this is for spin bowling analysis
    Returns:
        Dictionary of wrist flexion metrics
    """
    stats = joint_stats(wrist_angle_series)
    rom = range_of_motion(wrist_angle_series)
    
    metrics = {
        "wrist_flex_max": stats["max"],
        "wrist_flex_min": stats["min"],
        "wrist_flex_mean": stats["mean"],
        "wrist_flex_rom": rom
    }
    
    if is_spin:
        # Additional metrics specific to spin bowling
        series = pd.to_numeric(wrist_angle_series, errors='coerce')
        
        # Calculate rate of change for wrist angles
        if len(series) > 1:
            diff = np.diff(series)
            metrics["wrist_flex_rate"] = np.nanmean(np.abs(diff))
            metrics["max_flex_rate"] = np.nanmax(np.abs(diff))
        else:
            metrics["wrist_flex_rate"] = 0.0
            metrics["max_flex_rate"] = 0.0
            
    return metrics
