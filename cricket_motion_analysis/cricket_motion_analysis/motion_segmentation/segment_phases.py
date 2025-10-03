# Segment actions into key phases
from cricket_motion_analysis.feature_extraction.angular_velocity import angular__velocity

def segment_motion(processed_df, action, fps=30):
    """
    Returns a list of phase labels for each frame based on action/exercise.
    processed_df: DataFrame with smoothed joint angles
    action: string specifying action or exercise
    fps: video FPS
    """
    phases = ["unknown"] * len(processed_df)
    
    if action == "batting":
        elbow = processed_df["left_elbow"].values
        shoulder = processed_df["left_shoulder"].values
        vel_elbow = angular__velocity(elbow, fps)
        
        for i in range(len(elbow)):
            if elbow[i] > 150 and shoulder[i] > 160:
                phases[i] = "backswing"
            elif elbow[i] < 90 and vel_elbow[i] < 0:
                phases[i] = "downswing"
            else:
                phases[i] = "follow-through"
    
    elif action == "bowling":
        elbow = processed_df["right_elbow"].values
        knee = processed_df["right_knee"].values
        trunk = processed_df.get("trunk", np.zeros(len(processed_df)))
        vel_elbow = angular__velocity(elbow, fps)
        
        for i in range(len(elbow)):
            if knee[i] > 160 and trunk[i] < 10:
                phases[i] = "run-up"
            elif vel_elbow[i] > 50:
                phases[i] = "delivery"
            else:
                phases[i] = "follow-through"
    
    elif action == "throwing":
        elbow = processed_df["right_elbow"].values
        trunk = processed_df.get("trunk", np.zeros(len(processed_df)))
        vel_elbow = angular__velocity(elbow, fps)
        for i in range(len(elbow)):
            if elbow[i] > 150:
                phases[i] = "prep"
            elif vel_elbow[i] > 30:
                phases[i] = "throw"
            else:
                phases[i] = "follow-through"
    
    elif action in ["fielding", "wicket keeping"]:
        knee = processed_df["left_knee"].values
        trunk = processed_df.get("trunk", np.zeros(len(processed_df)))
        vel_knee = angular__velocity(knee, fps)
        for i in range(len(knee)):
            if knee[i] > 160:
                phases[i] = "ready"
            elif vel_knee[i] > 20:
                phases[i] = "movement"
            else:
                phases[i] = "recovery"
    
    elif action in ["push","pull","hinge","squat","lunge","explosive","mobility"]:
        # Generic gym segmentation
        # Use elbow for push/pull, knee for squat/lunge, trunk for hinge/explosive
        if action in ["push","pull"]:
            elbow = processed_df["left_elbow"].values
            vel_elbow = angular__velocity(elbow, fps)
            for i in range(len(elbow)):
                if vel_elbow[i] < -10:
                    phases[i] = "lowering"
                elif vel_elbow[i] > 10:
                    phases[i] = "lifting"
                else:
                    phases[i] = "hold"
        elif action in ["squat","lunge","hinge","explosive"]:
            knee = processed_df["left_knee"].values
            vel_knee = angular__velocity(knee, fps)
            for i in range(len(knee)):
                if vel_knee[i] < -5:
                    phases[i] = "eccentric"
                elif vel_knee[i] > 5:
                    phases[i] = "concentric"
                else:
                    phases[i] = "pause"
        elif action == "mobility":
            # Simple placeholder: based on motion direction of key joint
            joint = processed_df.get("left_hip", processed_df["left_knee"].values)
            vel_joint = angular__velocity(joint, fps)
            for i in range(len(joint)):
                if vel_joint[i] < -5:
                    phases[i] = "stretching"
                elif vel_joint[i] > 5:
                    phases[i] = "releasing"
                else:
                    phases[i] = "hold"
    
    return phases
