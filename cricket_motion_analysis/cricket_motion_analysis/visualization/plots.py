import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def plot_joint_angles_time_series(df: pd.DataFrame, joints: List[str], phases: bool = True):
    """
    Create an interactive time series plot of joint angles with phase overlays.
    
    Args:
        df: DataFrame containing joint angles and phases
        joints: List of joint names to plot
        phases: Whether to include phase overlays
    """
    fig = make_subplots(rows=len(joints), cols=1, shared_xaxes=True,
                        subplot_titles=joints)
    
    colors = sns.color_palette("husl", len(joints)).as_hex()
    
    for idx, joint in enumerate(joints, 1):
        fig.add_trace(
            go.Scatter(x=df['frame'], y=df[joint],
                      name=joint, line=dict(color=colors[idx-1])),
            row=idx, col=1
        )
        
        if phases and 'phase' in df.columns:
            # Add phase backgrounds
            phase_changes = df[df['phase'].shift() != df['phase']].index
            for start, end in zip(phase_changes[:-1], phase_changes[1:]):
                phase_name = df.loc[start, 'phase']
                fig.add_vrect(
                    x0=df.loc[start, 'frame'],
                    x1=df.loc[end, 'frame'],
                    fillcolor="gray",
                    opacity=0.1,
                    annotation_text=phase_name,
                    annotation_position="top left",
                    row=idx, col=1
                )
    
    fig.update_layout(
        height=300*len(joints),
        title="Joint Angles Over Time",
        showlegend=False,
        template="plotly_white"
    )
    
    return fig

def create_3d_skeleton_plot(landmarks: Dict[str, tuple], connections: List[tuple]):
    """
    Create an interactive 3D skeleton visualization.
    
    Args:
        landmarks: Dictionary of joint coordinates {joint_name: (x, y, z)}
        connections: List of tuples defining connections between joints
    """
    fig = go.Figure()
    
    # Plot joints as markers
    x = [coord[0] for coord in landmarks.values()]
    y = [coord[1] for coord in landmarks.values()]
    z = [coord[2] for coord in landmarks.values()]
    
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=6, color='blue'),
        name='Joints'
    ))
    
    # Plot connections between joints
    for start, end in connections:
        if start in landmarks and end in landmarks:
            fig.add_trace(go.Scatter3d(
                x=[landmarks[start][0], landmarks[end][0]],
                y=[landmarks[start][1], landmarks[end][1]],
                z=[landmarks[start][2], landmarks[end][2]],
                mode='lines',
                line=dict(color='red', width=2),
                name=f'{start}-{end}'
            ))
    
    fig.update_layout(
        scene=dict(
            camera=dict(up=dict(x=0, y=1, z=0)),
            aspectmode='data'
        ),
        title="3D Skeleton Visualization",
        showlegend=False
    )
    
    return fig

def plot_performance_radar(metrics: Dict[str, float]):
    """
    Create a radar plot of performance metrics.
    
    Args:
        metrics: Dictionary of performance metrics
    """
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Performance Metrics'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title="Performance Analysis"
    )
    
    return fig

def create_biomechanics_dashboard(processed_df: pd.DataFrame, 
                                performance_metrics: Dict[str, float],
                                risk_assessment: Dict[str, Dict],
                                detected_errors: Dict[str, List[Dict]] = None):
    """
    Create a comprehensive dashboard with all analysis visualizations.
    
    Args:
        processed_df: DataFrame with processed joint angles and phases
        performance_metrics: Dictionary of performance metrics
        risk_assessment: Dictionary of risk assessments by joint
        detected_errors: Dictionary of detected technique errors
    """
    # Create subplot figure
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=("Joint Angles", "Performance Metrics", 
                       "Risk Assessment", "Technique Errors",
                       "Movement Phases", "Joint Velocities"),
        specs=[[{"type": "xy"}, {"type": "polar"}],
               [{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]]
    )
    
    # Joint angles plot
    for joint in [col for col in processed_df.columns if col not in ['frame', 'phase']]:
        fig.add_trace(
            go.Scatter(x=processed_df['frame'], y=processed_df[joint],
                      name=joint),
            row=1, col=1
        )
    
    # Performance radar plot
    categories = list(performance_metrics.keys())
    values = list(performance_metrics.values())
    fig.add_trace(
        go.Scatterpolar(r=values, theta=categories, fill='toself'),
        row=1, col=2
    )
    
    # Risk assessment plot
    risk_scores = {joint: assessment['overall_risk_score'] 
                  for joint, assessment in risk_assessment.items()}
    fig.add_trace(
        go.Bar(x=list(risk_scores.keys()), y=list(risk_scores.values()),
               name="Risk Scores"),
        row=2, col=1
    )
    
    # Error timeline
    if detected_errors:
        for joint, errors in detected_errors.items():
            for error in errors:
                fig.add_trace(
                    go.Scatter(x=[error['start_frame'], error['end_frame']],
                              y=[error['max_deviation'], error['max_deviation']],
                              name=f"{joint} error"),
                    row=2, col=2
                )
    
    # Phase distribution
    if 'phase' in processed_df.columns:
        phase_counts = processed_df['phase'].value_counts()
        fig.add_trace(
            go.Bar(x=phase_counts.index, y=phase_counts.values,
                   name="Movement Phases"),
            row=3, col=1
        )
    
    # Joint velocities
    for joint in [col for col in processed_df.columns if col not in ['frame', 'phase']]:
        velocities = np.gradient(processed_df[joint])
        fig.add_trace(
            go.Scatter(x=processed_df['frame'], y=velocities,
                      name=f"{joint} velocity"),
            row=3, col=2
        )
    
    fig.update_layout(
        height=1200,
        width=1600,
        title_text="Biomechanics Analysis Dashboard",
        showlegend=True
    )
    
    return fig

def save_analysis_report(output_path: str,
                        processed_df: pd.DataFrame,
                        performance_metrics: Dict[str, float],
                        risk_assessment: Dict[str, Dict],
                        detected_errors: Dict[str, List[Dict]],
                        action_type: str):
    """
    Generate and save a comprehensive HTML report with all visualizations.
    
    Args:
        output_path: Path to save the HTML report
        processed_df: DataFrame with processed joint angles and phases
        performance_metrics: Dictionary of performance metrics
        risk_assessment: Dictionary of risk assessments by joint
        detected_errors: Dictionary of detected technique errors
        action_type: Type of movement being analyzed
    """
    # Create all visualizations
    time_series = plot_joint_angles_time_series(
        processed_df,
        [col for col in processed_df.columns if col not in ['frame', 'phase']]
    )
    
    performance_radar = plot_performance_radar(performance_metrics)
    
    dashboard = create_biomechanics_dashboard(
        processed_df,
        performance_metrics,
        risk_assessment,
        detected_errors
    )
    
    # Generate HTML report
    html_content = f"""
    <html>
    <head>
        <title>Biomechanics Analysis Report - {action_type}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>Biomechanics Analysis Report</h1>
        <h2>Movement Type: {action_type}</h2>
        
        <h3>Time Series Analysis</h3>
        <div id="time-series"></div>
        
        <h3>Performance Metrics</h3>
        <div id="performance-radar"></div>
        
        <h3>Comprehensive Dashboard</h3>
        <div id="dashboard"></div>
        
        <script>
            {time_series.to_json()}
            {performance_radar.to_json()}
            {dashboard.to_json()}
        </script>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
