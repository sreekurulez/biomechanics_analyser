import numpy as np
from typing import Dict, Union

def normalize_by_height(data: np.ndarray, player_height: float, reference_height: float = 1.75) -> np.ndarray:
    """
    Normalize joint angles based on player height ratio.
    
    Args:
        data: Joint angle data to normalize
        player_height: Height of the player in meters
        reference_height: Reference height (default 1.75m - average adult male height)
    
    Returns:
        Normalized joint angle data
    """
    height_ratio = reference_height / player_height
    # Scale only displacement-based measurements, not angular measurements
    return data * height_ratio if np.max(np.abs(data)) > 360 else data

def normalize_by_limb_length(data: Dict[str, np.ndarray], limb_lengths: Dict[str, float], 
                           reference_lengths: Dict[str, float]) -> Dict[str, np.ndarray]:
    """
    Normalize joint movements based on individual limb lengths.
    
    Args:
        data: Dictionary of joint angle data
        limb_lengths: Dictionary of player's limb lengths
        reference_lengths: Dictionary of reference limb lengths
    
    Returns:
        Dictionary of normalized joint angle data
    """
    normalized_data = {}
    for joint, angles in data.items():
        if joint in limb_lengths and joint in reference_lengths:
            length_ratio = reference_lengths[joint] / limb_lengths[joint]
            # Apply normalization only to relevant measurements
            normalized_data[joint] = angles * length_ratio if np.max(np.abs(angles)) > 360 else angles
        else:
            normalized_data[joint] = angles
    return normalized_data

def calculate_bmi_factor(weight: float, height: float) -> float:
    """
    Calculate BMI-based adjustment factor for movement analysis.
    
    Args:
        weight: Weight in kilograms
        height: Height in meters
    
    Returns:
        BMI adjustment factor
    """
    bmi = weight / (height * height)
    # Create an adjustment factor that varies less dramatically than raw BMI
    return np.clip(1 + (bmi - 22) * 0.01, 0.9, 1.1)  # 22 is considered optimal BMI for athletes

def adjust_thresholds_by_anthropometrics(thresholds: Dict[str, float], 
                                       height: float, 
                                       weight: float,
                                       exercise_type: str = None) -> Dict[str, float]:
    """
    Adjust movement thresholds based on player's anthropometric data.
    
    Args:
        thresholds: Dictionary of movement thresholds
        height: Player height in meters
        weight: Player weight in kilograms
        exercise_type: Type of exercise (push, pull, hinge, squat, etc.)
    
    Returns:
        Dictionary of adjusted thresholds
    """
    bmi_factor = calculate_bmi_factor(weight, height)
    height_factor = height / 1.75  # Relative to average height
    
    adjusted_thresholds = {}
    
    # Exercise-specific anthropometric adjustments
    exercise_factors = {
        'squat': {
            'rom_factor': min(1.0, height / 1.75),  # Taller people might need less deep squat
            'knee_angle': 1.0 + (height - 1.75) * 0.1  # Adjust knee angle expectations
        },
        'hinge': {
            'hip_factor': height / 1.75,  # Adjust hip hinge expectations for height
            'spine_angle': 1.0 + (height - 1.75) * 0.05  # Slight spine angle adjustment
        },
        'push': {
            'arm_factor': height_factor,  # Adjust for arm length in push movements
            'shoulder_width': bmi_factor  # Consider shoulder width for form
        },
        'pull': {
            'arm_factor': height_factor,  # Adjust for arm length in pull movements
            'scapular_factor': bmi_factor  # Consider build for scapular movement
        },
        'lunge': {
            'stride_factor': height_factor,  # Adjust stride length expectations
            'knee_tracking': 1.0 + (height - 1.75) * 0.08  # Adjust knee tracking thresholds
        }
    }
    
    for metric, value in thresholds.items():
        # Base adjustments
        if 'velocity' in metric.lower():
            adjusted_value = value * height_factor
        elif 'force' in metric.lower() or 'power' in metric.lower():
            adjusted_value = value * bmi_factor
        else:
            adjusted_value = value
            
        # Exercise-specific adjustments
        if exercise_type and exercise_type in exercise_factors:
            factors = exercise_factors[exercise_type]
            
            if 'squat' in exercise_type:
                if 'depth' in metric.lower():
                    adjusted_value *= factors['rom_factor']
                elif 'knee' in metric.lower():
                    adjusted_value *= factors['knee_angle']
                    
            elif 'hinge' in exercise_type:
                if 'hip' in metric.lower():
                    adjusted_value *= factors['hip_factor']
                elif 'spine' in metric.lower():
                    adjusted_value *= factors['spine_angle']
                    
            elif 'push' in exercise_type:
                if 'elbow' in metric.lower() or 'shoulder' in metric.lower():
                    adjusted_value *= factors['arm_factor']
                    
            elif 'pull' in exercise_type:
                if 'back' in metric.lower() or 'lat' in metric.lower():
                    adjusted_value *= factors['arm_factor']
                    
            elif 'lunge' in exercise_type:
                if 'stride' in metric.lower():
                    adjusted_value *= factors['stride_factor']
                elif 'knee' in metric.lower():
                    adjusted_value *= factors['knee_tracking']
                    
        adjusted_thresholds[metric] = adjusted_value
            
    return adjusted_thresholds

def calculate_exercise_rom_limits(height: float, weight: float, exercise_type: str) -> Dict[str, Dict[str, float]]:
    """
    Calculate exercise-specific range of motion limits based on anthropometrics.
    
    Args:
        height: Player height in meters
        weight: Player weight in kilograms
        exercise_type: Type of exercise
        
    Returns:
        Dictionary of ROM limits for different joints
    """
    bmi = weight / (height * height)
    
    # Base ROM limits
    base_limits = {
        'squat': {
            'hip': {'min': 0, 'max': 120},
            'knee': {'min': 0, 'max': 140},
            'ankle': {'min': -20, 'max': 30}
        },
        'hinge': {
            'hip': {'min': 0, 'max': 90},
            'spine': {'min': -15, 'max': 15},
            'knee': {'min': 0, 'max': 20}
        },
        'push': {
            'shoulder': {'min': -30, 'max': 180},
            'elbow': {'min': 0, 'max': 160}
        },
        'pull': {
            'shoulder': {'min': -60, 'max': 180},
            'elbow': {'min': 0, 'max': 160},
            'scapula': {'min': -30, 'max': 30}
        }
    }
    
    if exercise_type not in base_limits:
        return {}
        
    adjusted_limits = {}
    limits = base_limits[exercise_type]
    
    # Adjust limits based on anthropometrics
    for joint, angles in limits.items():
        adjusted_limits[joint] = {
            'min': angles['min'],
            'max': angles['max']
        }
        
        # Height-based adjustments
        if height > 1.85:  # For taller individuals
            if joint in ['knee', 'hip'] and exercise_type == 'squat':
                # Reduce required ROM for very tall individuals
                adjusted_limits[joint]['max'] *= 0.9
            elif joint == 'spine' and exercise_type == 'hinge':
                # Adjust spine angles for taller people
                adjusted_limits[joint]['max'] *= 1.1
                
        # Weight/BMI-based adjustments
        if bmi > 25:
            if joint in ['hip', 'knee']:
                # Adjust ROM expectations for higher BMI
                adjusted_limits[joint]['max'] *= 0.95
                
    return adjusted_limits