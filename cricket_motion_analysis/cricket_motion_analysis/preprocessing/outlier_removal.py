import numpy as np
import pandas as pd

def remove_outliers_zscore(series, z_thresh=3):
    """Replace outliers using Z-score method."""
    mean, std = np.mean(series), np.std(series)
    cleaned = []
    for val in series:
        if abs((val - mean) / std) > z_thresh:
            cleaned.append(np.nan)
        else:
            cleaned.append(val)
    return cleaned

def remove_outliers_iqr(series, factor=1.5):
    """Replace outliers using IQR method."""
    q1, q3 = np.percentile(series, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - factor*iqr, q3 + factor*iqr
    return [val if (lower <= val <= upper) else np.nan for val in series]

def interpolate_missing(series):
    """Fill NaNs by linear interpolation."""
    s = pd.Series(series)
    return s.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').tolist()

