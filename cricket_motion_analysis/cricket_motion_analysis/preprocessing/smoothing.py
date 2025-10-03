import numpy as np
from scipy.signal import savgol_filter

def moving_average(series, window_size=5):
    """Apply moving average smoothing."""
    if len(series) < window_size:
        return series
    return np.convolve(series, np.ones(window_size)/window_size, mode='same')

def savitzky_golay(series, window_size=5, poly_order=2):
    """Apply Savitzky-Golay filter for smooth + sharp features."""
    if len(series) < window_size:
        return series
    return savgol_filter(series, window_size, poly_order)

def kalman_filter(series, process_var=1e-5, measurement_var=1e-2):
    """Simple 1D Kalman filter for smoothing."""
    n = len(series)
    xhat = np.zeros(n)      # a posteri estimate of x
    P = np.zeros(n)         # a posteri error estimate
    xhat[0] = series[0]
    P[0] = 1.0
    Q = process_var
    R = measurement_var

    for k in range(1, n):
        # Predict
        xhatminus = xhat[k-1]
        Pminus = P[k-1] + Q

        # Update
        K = Pminus / (Pminus + R)
        xhat[k] = xhatminus + K * (series[k] - xhatminus)
        P[k] = (1 - K) * Pminus

    return xhat

