import numpy as np

def calculate_angle(a, b, c):
	"""Calculate the angle at point b given three points a, b, c."""
	a, b, c = np.array(a), np.array(b), np.array(c)
	ba = a - b
	bc = c - b
	cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
	angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
	return angle

def calculate_trunk_angle(left_shoulder, right_shoulder, left_hip, right_hip):
    """Calculate trunk tilt angle between shoulders and hips."""
    shoulder_mid = ((left_shoulder[0]+right_shoulder[0])//2, (left_shoulder[1]+right_shoulder[1])//2)
    hip_mid = ((left_hip[0]+right_hip[0])//2, (left_hip[1]+right_hip[1])//2)
    dx = shoulder_mid[0]-hip_mid[0]
    dy = shoulder_mid[1]-hip_mid[1]
    trunk_angle = np.degrees(np.arctan2(dy,dx))
    return trunk_angle
