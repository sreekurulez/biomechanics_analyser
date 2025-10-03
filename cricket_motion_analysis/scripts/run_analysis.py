import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cricket_motion_analysis.utils.angle_utils import calculate_angle, calculate_trunk_angle
from cricket_motion_analysis.utils.pose_utils import get_joints_for_action, get_pose_estimator, get_landmarks, get_coords
from cricket_motion_analysis.utils.video_io import open_video, create_writer
from cricket_motion_analysis.preprocessing.outlier_removal import remove_outliers_zscore, interpolate_missing
from cricket_motion_analysis.preprocessing.normalize import normalize_time
from cricket_motion_analysis.preprocessing.anthropometric import calculate_bmi_factor
from scipy.signal import savgol_filter as savitzky_golay
from cricket_motion_analysis.feature_extraction import extract_features
from cricket_motion_analysis.motion_segmentation.segment_phases import segment_motion
from cricket_motion_analysis.analysis.performance_analysis import BiomechanicsAnalyzer

def main():
    # -----------------
    # User Input
    # -----------------
    print("Enter player measurements:")
    player_height = float(input("Height (in meters): ").strip() or "1.75")
    player_weight = float(input("Weight (in kg): ").strip() or "75")
    
    print("\nEnter limb measurements (press Enter to use default):")
    player_limb_lengths = {
        'arm': float(input("Arm length (shoulder to wrist in meters): ").strip() or "0.73"),
        'leg': float(input("Leg length (hip to ankle in meters): ").strip() or "0.85"),
        'torso': float(input("Torso length (shoulder to hip in meters): ").strip() or "0.52")
    }
    
    print("\nSelect action type:")
    print("1. Batting\n2. Bowling\n3. Throwing\n4. Fielding\n5. Wicket Keeping\n6. Gym Exercise")
    action_input = input("Enter action name or number: ").strip()

    if action_input == "6" or action_input.lower() == "gym exercise":
        print("Select gym exercise type:")
        print("1. Push\n2. Pull\n3. Hinge\n4. Squat\n5. Lunge\n6. Explosive / Plyometric\n7. Mobility / Flexibility")
        gym_input = input("Enter gym exercise name or number: ").strip()
        gym_map = {"1":"push","2":"pull","3":"hinge","4":"squat","5":"lunge","6":"explosive","7":"mobility"}
        action = gym_map.get(gym_input, gym_input.lower())
        subtype = None
    else:
        action_map = {"1":"batting","2":"bowling","3":"throwing","4":"fielding","5":"wicket keeping"}
        action = action_map.get(action_input, action_input.lower())
        
        # Get specific type for batting or bowling
        if action == "batting":
            print("\nSelect shot type:")
            print("1. Straight Drive")
            print("2. Cover Drive")
            print("3. Pull Shot")
            print("4. Cut Shot")
            print("5. Front-foot Defense")
            print("6. Back-foot Defense")
            shot_map = {
                "1": "straight_drive",
                "2": "cover_drive",
                "3": "pull_shot",
                "4": "cut_shot",
                "5": "front_foot_defense",
                "6": "back_foot_defense"
            }
            shot_input = input("Enter shot type number: ").strip()
            subtype = shot_map.get(shot_input, "straight_drive")
        elif action == "bowling":
            print("\nSelect bowling type:")
            print("1. Fast Bowling")
            print("2. Off Spin")
            print("3. Leg Spin")
            print("4. Medium Pace")
            bowling_map = {
                "1": "fast_bowling",
                "2": "off_spin",
                "3": "leg_spin",
                "4": "medium_pace"
            }
            bowling_input = input("Enter bowling type number: ").strip()
            subtype = bowling_map.get(bowling_input, "fast_bowling")
        else:
            subtype = None
    print(f"Analyzing: {action}")

    input_video_path = input("Enter path to input video (default: ./data/raw_videos/IMG_9383.MOV): ").strip() or "./data/raw_videos/IMG_9383.MOV"
    output_video_path = f"./outputs/processed_{action}.mp4"
    csv_output_path = f"./outputs/joint_angles_{action}_raw.csv"
    os.makedirs("outputs", exist_ok=True)

    if not os.path.exists(input_video_path):
        print(f"Input video not found: {input_video_path}")
        return

    # -----------------
    # Capture Video
    # -----------------
    cap, frame_width, frame_height, fps = open_video(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = create_writer(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Pose Estimator
    pose, mp_pose = get_pose_estimator()
    joints_to_track = get_joints_for_action(action)
    joint_angles = {"frame": []}
    for joint_name in joints_to_track.keys():
        joint_angles[joint_name] = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        lm = get_landmarks(results)
        h, w, _ = frame.shape
        if lm:
            for joint_name, points in joints_to_track.items():
                if joint_name != "trunk":
                    a = get_coords(lm, points[0], mp_pose, w, h)
                    b = get_coords(lm, points[1], mp_pose, w, h)
                    c = get_coords(lm, points[2], mp_pose, w, h)
                    angle = calculate_angle(a, b, c)
                    joint_angles[joint_name].append(angle)
                    cv2.putText(frame, f"{joint_name}: {int(angle)}°", (b[0]+10,b[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
                else:
                    left_shoulder = get_coords(lm, points[0], mp_pose, w, h)
                    right_shoulder = get_coords(lm, points[1], mp_pose, w, h)
                    left_hip = get_coords(lm, points[2], mp_pose, w, h)
                    right_hip = get_coords(lm, points[3], mp_pose, w, h)
                    trunk_angle = calculate_trunk_angle(left_shoulder, right_shoulder, left_hip, right_hip)
                    joint_angles[joint_name].append(trunk_angle)
                    hip_mid = ((left_hip[0]+right_hip[0])//2, (left_hip[1]+right_hip[1])//2)
                    cv2.putText(frame, f"{joint_name}: {int(trunk_angle)}°", (hip_mid[0]+10, hip_mid[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            joint_angles["frame"].append(frame_count)
        out.write(frame)
        cv2.imshow("Pose Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # -----------------
    # Save raw joint angles CSV
    # -----------------
    df = pd.DataFrame(joint_angles)
    df.to_csv(csv_output_path, index=False)
    print(f"Raw joint angles saved to {csv_output_path}")

    # -----------------
    # Preprocessing Step
    # -----------------
    processed_df = df.copy()
    for joint_name in joints_to_track.keys():
        cleaned = remove_outliers_zscore(processed_df[joint_name].values, z_thresh=3)
        cleaned = interpolate_missing(cleaned)
        smoothed = savitzky_golay(cleaned, window_length=7, polyorder=2)
        normalized = normalize_time(smoothed, target_len=len(processed_df))
        processed_df[joint_name] = normalized
    processed_df["frame"] = list(range(1, len(processed_df)+1))

    # -----------------
    # Motion Segmentation
    # -----------------
    processed_df["phase"] = segment_motion(processed_df, action, fps=fps)

    # -----------------
    # Save processed CSV with phase
    # -----------------
    processed_csv_path = f"./outputs/joint_angles_{action}_processed_with_phase.csv"
    processed_df.to_csv(processed_csv_path, index=False)
    print(f"Processed joint angles with phase saved to {processed_csv_path}")

    # -----------------
    # Extract features
    # -----------------
    features = extract_features(processed_df, fps=fps)
    features_df = features.to_frame().T
    features_df.to_csv(f"./outputs/features_{action}.csv", index=False)

    # -----------------
    # Enhanced Visualizations
    # -----------------
    from cricket_motion_analysis.visualization.plots import (
        plot_joint_angles_time_series,
        create_3d_skeleton_plot,
        plot_performance_radar,
        create_biomechanics_dashboard,
        save_analysis_report
    )
    from cricket_motion_analysis.visualization.video_overlay import (
        create_analysis_video,
        create_3d_visualization_video
    )
    
    print("\nGenerating visualizations...")
    
    # Create interactive time series plot
    time_series_fig = plot_joint_angles_time_series(
        processed_df,
        list(joints_to_track.keys())
    )
    time_series_fig.write_html(f"./outputs/time_series_{action}.html")
    
    # Create performance radar plot
    if 'performance_metrics' in locals():
        radar_fig = plot_performance_radar(performance_metrics)
        radar_fig.write_html(f"./outputs/performance_radar_{action}.html")
    
    # Create comprehensive dashboard
    if all(var in locals() for var in ['processed_df', 'performance_metrics', 'risk_assessment', 'errors']):
        dashboard_fig = create_biomechanics_dashboard(
            processed_df,
            performance_metrics,
            risk_assessment,
            errors
        )
        dashboard_fig.write_html(f"./outputs/dashboard_{action}.html")
        
    # Generate comprehensive HTML report
    save_analysis_report(
        f"./outputs/analysis_report_{action}.html",
        processed_df,
        performance_metrics if 'performance_metrics' in locals() else {},
        risk_assessment if 'risk_assessment' in locals() else {},
        errors if 'errors' in locals() else {},
        action
    )
    
    # Create enhanced video analysis
    if results and lm:  # If we have pose estimation results
        landmark_series = []
        connections = []
        
        # Extract landmark series and connections from pose estimation
        # This part needs to be adapted based on your pose estimation data structure
        
        # Create analysis video with overlays
        create_analysis_video(
            input_video_path,
            f"./outputs/analysis_video_{action}.mp4",
            processed_df,
            landmark_series,
            connections,
            performance_metrics if 'performance_metrics' in locals() else None
        )
        
        # Create 3D visualization if 3D data is available
        if hasattr(results, 'pose_world_landmarks') and results.pose_world_landmarks:
            create_3d_visualization_video(
                landmark_series,  # Need to convert to 3D coordinates
                connections,
                f"./outputs/3d_visualization_{action}.mp4"
            )
            
    print("\nVisualization files generated in ./outputs/ directory")

    # -----------------
    # Overlay phases on processed video
    # -----------------
    cap, _, _, _ = open_video(input_video_path)
    out = create_writer(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # Add phase overlay
        phase_text = processed_df["phase"].iloc[frame_count-1]
        cv2.putText(frame, f"Phase: {phase_text}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 3)
        out.write(frame)
    cap.release()
    out.release()
    print(f"Processed video with phase overlay saved to {output_video_path}")

    # -----------------
    # Biomechanics Analysis
    # -----------------
    print("\nPerforming Biomechanics Analysis...")
    analyzer = BiomechanicsAnalyzer()
    
    # Load or set reference patterns (example with expert performance data)
    if subtype:
        reference_data_path = f"./data/reference_patterns/{action}/{subtype}_reference.csv"
    else:
        reference_data_path = f"./data/reference_patterns/{action}_reference.csv"
    if os.path.exists(reference_data_path):
        reference_df = pd.read_csv(reference_data_path)
        # Create reference patterns dictionary
        reference_patterns = {joint: reference_df[joint].values for joint in joints_to_track.keys()}
        
        # Normalize reference patterns to match the length of captured data
        from cricket_motion_analysis.preprocessing.normalize import normalize_reference_pattern
        normalized_reference = normalize_reference_pattern(reference_patterns, len(processed_df))
        analyzer.set_reference_pattern(action, normalized_reference)
        
        # Compare with reference patterns, using anthropometric data
        current_patterns = {joint: processed_df[joint].values for joint in joints_to_track.keys()}
        similarity, joint_similarities, _ = analyzer.compare_with_reference(
            current_patterns, 
            action,
            player_height=player_height,
            player_weight=player_weight,
            player_limb_lengths=player_limb_lengths
        )
        print(f"\nTechnique Analysis:")
        print(f"Overall similarity to reference: {similarity:.2%}")
        print("\nJoint-wise similarity scores:")
        for joint, score in joint_similarities.items():
            print(f"{joint}: {score:.2%}")
            
        print("\nAnthropometric Adjustments Applied:")
        print(f"Height Ratio: {player_height/1.75:.2f}")
        print(f"BMI Factor: {calculate_bmi_factor(player_weight, player_height):.2f}")
        
        # Detect technique errors
        errors = analyzer.detect_technique_errors(current_patterns, action)
        if errors:
            print("\nDetected Technique Errors:")
            for joint, joint_errors in errors.items():
                for error in joint_errors:
                    print(f"{joint}: Deviation at frames {error['start_frame']}-{error['end_frame']} "
                          f"(max deviation: {error['max_deviation']:.2f}°)")
    
    # Calculate joint velocities for injury risk assessment
    joint_velocities = {}
    for joint in joints_to_track.keys():
        angles = processed_df[joint].values
        velocities = np.gradient(angles) * fps  # degrees per second
        joint_velocities[joint] = velocities
    
    # Assess injury risks
    risk_assessment = analyzer.assess_injury_risk(
        {joint: processed_df[joint].values for joint in joints_to_track.keys()},
        joint_velocities
    )
    
    print("\nInjury Risk Assessment:")
    for joint, assessment in risk_assessment.items():
        print(f"\n{joint.title()}:")
        print(f"  Overall Risk Score: {assessment['overall_risk_score']:.1f}/100")
        print(f"  Angle Violations: {assessment['angle_violation_percentage']:.1f}%")
        print(f"  Velocity Violations: {assessment['velocity_violation_percentage']:.1f}%")
        print(f"  Max Velocity: {assessment['max_velocity']:.1f}°/s")
    
    # Calculate performance metrics
    performance_metrics = analyzer.calculate_performance_metrics(
        {joint: processed_df[joint].values for joint in joints_to_track.keys()},
        action
    )
    
    print("\nPerformance Metrics:")
    print(f"Overall Efficiency Score: {performance_metrics['efficiency_score']:.2%}")
    print(f"Technique Similarity: {performance_metrics['overall_technique_similarity']:.2%}")
    print(f"Movement Symmetry: {performance_metrics['overall_symmetry']:.2%}")
    print(f"Movement Smoothness: {performance_metrics['movement_smoothness']:.2%}")
    print(f"Technique Consistency: {performance_metrics['technique_consistency']:.2%}")
    
    # Save analysis results
    analysis_results = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "performance_metrics": performance_metrics,
        "risk_assessment": risk_assessment,
        "technique_similarity": similarity if 'similarity' in locals() else None,
        "detected_errors": errors if 'errors' in locals() else None
    }
    
    analysis_output_path = f"./outputs/biomechanics_analysis_{action}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(analysis_output_path, 'w') as f:
        import json
        json.dump(analysis_results, f, indent=4)
    print(f"\nAnalysis results saved to {analysis_output_path}")


if __name__ == "__main__":
    main()
