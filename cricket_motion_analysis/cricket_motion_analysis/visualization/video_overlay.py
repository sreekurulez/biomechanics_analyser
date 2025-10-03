import cv2
import numpy as np
from typing import Dict, List, Tuple
import mediapipe as mp

def create_skeleton_overlay(frame: np.ndarray, 
                          landmarks: Dict[str, Tuple[int, int]], 
                          connections: List[Tuple[str, str]],
                          angles: Dict[str, float] = None):
    """
    Draw skeleton overlay with joint angles on frame.
    
    Args:
        frame: Video frame
        landmarks: Dictionary of joint coordinates
        connections: List of joint connections to draw
        angles: Optional dictionary of joint angles to display
    """
    overlay = frame.copy()
    
    # Draw connections
    for start, end in connections:
        if start in landmarks and end in landmarks:
            start_point = landmarks[start]
            end_point = landmarks[end]
            cv2.line(overlay, start_point, end_point, (0, 255, 0), 2)
    
    # Draw joints
    for joint, pos in landmarks.items():
        cv2.circle(overlay, pos, 5, (255, 0, 0), -1)
        
        # Add angle labels if provided
        if angles and joint in angles:
            cv2.putText(overlay, 
                       f"{joint}: {angles[joint]:.1f}°",
                       (pos[0] + 10, pos[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (0, 0, 255),
                       2)
    
    return overlay

def add_phase_indicator(frame: np.ndarray, 
                       phase: str,
                       position: Tuple[int, int] = (50, 50)):
    """
    Add movement phase indicator to frame.
    
    Args:
        frame: Video frame
        phase: Current movement phase
        position: Position to place the text
    """
    cv2.putText(frame,
                f"Phase: {phase}",
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2)
    return frame

def add_performance_metrics(frame: np.ndarray,
                          metrics: Dict[str, float],
                          start_position: Tuple[int, int] = (50, 100)):
    """
    Add real-time performance metrics to frame.
    
    Args:
        frame: Video frame
        metrics: Dictionary of performance metrics
        start_position: Starting position for metrics display
    """
    y_position = start_position[1]
    for metric, value in metrics.items():
        cv2.putText(frame,
                    f"{metric}: {value:.2f}",
                    (start_position[0], y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)
        y_position += 30
    return frame

def create_analysis_video(input_path: str,
                         output_path: str,
                         processed_df: Dict[str, np.ndarray],
                         landmarks_series: List[Dict[str, Tuple[int, int]]],
                         connections: List[Tuple[str, str]],
                         performance_metrics: Dict[str, float] = None):
    """
    Create video with overlaid biomechanics analysis.
    
    Args:
        input_path: Path to input video
        output_path: Path for output video
        processed_df: Dictionary of processed joint angles
        landmarks_series: List of landmark coordinates for each frame
        connections: List of joint connections
        performance_metrics: Optional dictionary of performance metrics
    """
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Add skeleton overlay
        if frame_idx < len(landmarks_series):
            frame = create_skeleton_overlay(
                frame,
                landmarks_series[frame_idx],
                connections,
                {joint: angles[frame_idx] for joint, angles in processed_df.items()
                 if joint not in ['frame', 'phase']}
            )
        
        # Add phase if available
        if 'phase' in processed_df:
            frame = add_phase_indicator(frame, processed_df['phase'][frame_idx])
        
        # Add performance metrics if available
        if performance_metrics:
            frame = add_performance_metrics(frame, performance_metrics)
        
        out.write(frame)
        frame_idx += 1
        
        # Display progress
        if frame_idx % 30 == 0:
            print(f"Processing frame {frame_idx}")
    
    cap.release()
    out.release()
    print(f"Analysis video saved to {output_path}")

def create_3d_visualization_video(landmarks_series: List[Dict[str, Tuple[float, float, float]]],
                                connections: List[Tuple[str, str]],
                                output_path: str,
                                fps: int = 30):
    """
    Create a rotating 3D visualization video of the movement.
    
    Args:
        landmarks_series: List of 3D landmark coordinates for each frame
        connections: List of joint connections
        output_path: Path for output video
        fps: Frames per second for output video
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create figure for 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up video writer
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (800, 800)  # Output size
    )
    
    # Calculate bounds for consistent view
    all_x = []
    all_y = []
    all_z = []
    for landmarks in landmarks_series:
        for coord in landmarks.values():
            all_x.append(coord[0])
            all_y.append(coord[1])
            all_z.append(coord[2])
    
    x_range = (min(all_x), max(all_x))
    y_range = (min(all_y), max(all_y))
    z_range = (min(all_z), max(all_z))
    
    # Create frames
    rotation_angle = 0
    for frame_idx, landmarks in enumerate(landmarks_series):
        ax.clear()
        
        # Plot joints
        x = [coord[0] for coord in landmarks.values()]
        y = [coord[1] for coord in landmarks.values()]
        z = [coord[2] for coord in landmarks.values()]
        ax.scatter(x, y, z, c='blue', marker='o')
        
        # Plot connections
        for start, end in connections:
            if start in landmarks and end in landmarks:
                ax.plot([landmarks[start][0], landmarks[end][0]],
                       [landmarks[start][1], landmarks[end][1]],
                       [landmarks[start][2], landmarks[end][2]],
                       'r-')
        
        # Set consistent view limits
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        
        # Rotate view
        ax.view_init(elev=30, azim=rotation_angle)
        rotation_angle += 2
        
        # Convert plot to image
        plt.savefig('temp_frame.png')
        frame = cv2.imread('temp_frame.png')
        writer.write(frame)
        
        # Display progress
        if frame_idx % 10 == 0:
            print(f"Processing 3D frame {frame_idx}")
    
    writer.release()
    print(f"3D visualization video saved to {output_path}")
