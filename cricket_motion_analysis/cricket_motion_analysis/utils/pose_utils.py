import mediapipe as mp

def get_pose_estimator():
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5), mp_pose

def get_landmarks(results):
    if results.pose_landmarks:
        return results.pose_landmarks.landmark
    return None

def get_coords(lm, idx_name, mp_pose, w, h):
    idx = mp_pose.PoseLandmark[idx_name].value
    return (int(lm[idx].x * w), int(lm[idx].y * h))

def get_joints_for_action(action):
	"""Return joint definitions for a given action type."""
	joints = {}
	if action.lower() == "batting":
		joints = {
			"left_elbow": ["LEFT_SHOULDER","LEFT_ELBOW","LEFT_WRIST"],
			"right_elbow": ["RIGHT_SHOULDER","RIGHT_ELBOW","RIGHT_WRIST"],
			"left_knee": ["LEFT_HIP","LEFT_KNEE","LEFT_ANKLE"],
			"right_knee": ["RIGHT_HIP","RIGHT_KNEE","RIGHT_ANKLE"]
		}
	elif action.lower() == "bowling":
		joints = {
			"left_elbow": ["LEFT_SHOULDER","LEFT_ELBOW","LEFT_WRIST"],
			"right_elbow": ["RIGHT_SHOULDER","RIGHT_ELBOW","RIGHT_WRIST"],
			"left_knee": ["LEFT_HIP","LEFT_KNEE","LEFT_ANKLE"],
			"right_knee": ["RIGHT_HIP","RIGHT_KNEE","RIGHT_ANKLE"],
			"trunk": ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_HIP","RIGHT_HIP"]
		}
	elif action.lower() == "throwing":
		joints = {
			"left_elbow": ["LEFT_SHOULDER","LEFT_ELBOW","LEFT_WRIST"],
			"right_elbow": ["RIGHT_SHOULDER","RIGHT_ELBOW","RIGHT_WRIST"],
			"trunk": ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_HIP","RIGHT_HIP"]
		}
	elif action.lower() == "fielding":
		joints = {
			"left_knee": ["LEFT_HIP","LEFT_KNEE","LEFT_ANKLE"],
			"right_knee": ["RIGHT_HIP","RIGHT_KNEE","RIGHT_ANKLE"]
		}
	elif action.lower() == "wicket keeping":
		joints = {
			"left_knee": ["LEFT_HIP","LEFT_KNEE","LEFT_ANKLE"],
			"right_knee": ["RIGHT_HIP","RIGHT_KNEE","RIGHT_ANKLE"],
			"trunk": ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_HIP","RIGHT_HIP"]
		}
	elif action.lower() in ["push","pull","hinge","squat","lunge","explosive","mobility"]:
		# Generic gym movement
		if action.lower() in ["push"]:
			joints = {
				"left_elbow": ["LEFT_SHOULDER","LEFT_ELBOW","LEFT_WRIST"],
				"right_elbow": ["RIGHT_SHOULDER","RIGHT_ELBOW","RIGHT_WRIST"]
			}
		elif action.lower() in ["pull"]:
			joints = {
				"left_elbow": ["LEFT_SHOULDER","LEFT_ELBOW","LEFT_WRIST"],
				"right_elbow": ["RIGHT_SHOULDER","RIGHT_ELBOW","RIGHT_WRIST"]
			}
		elif action.lower() in ["hinge"]:
			joints = {
				"left_knee": ["LEFT_HIP","LEFT_KNEE","LEFT_ANKLE"],
				"right_knee": ["RIGHT_HIP","RIGHT_KNEE","RIGHT_ANKLE"],
				"trunk": ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_HIP","RIGHT_HIP"]
			}
		elif action.lower() in ["squat","lunge"]:
			joints = {
				"left_knee": ["LEFT_HIP","LEFT_KNEE","LEFT_ANKLE"],
				"right_knee": ["RIGHT_HIP","RIGHT_KNEE","RIGHT_ANKLE"],
				"trunk": ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_HIP","RIGHT_HIP"]
			}
		elif action.lower() in ["explosive","mobility"]:
			joints = {
				"left_elbow": ["LEFT_SHOULDER","LEFT_ELBOW","LEFT_WRIST"],
				"right_elbow": ["RIGHT_SHOULDER","RIGHT_ELBOW","RIGHT_WRIST"],
				"left_knee": ["LEFT_HIP","LEFT_KNEE","LEFT_ANKLE"],
				"right_knee": ["RIGHT_HIP","RIGHT_KNEE","RIGHT_ANKLE"],
				"trunk": ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_HIP","RIGHT_HIP"]
			}
	return joints
