import numpy as np
from typing import Dict, List, Tuple
from ..feature_extraction.symmetry_speed import calculate_symmetry_score
from fastdtw import fastdtw
from scipy.stats import pearsonr

from ..preprocessing.anthropometric import (
    normalize_by_height,
    normalize_by_limb_length,
    adjust_thresholds_by_anthropometrics,
    calculate_bmi_factor
)

class BiomechanicsAnalyzer:
    def __init__(self):
        self.reference_patterns = {}
        self.reference_anthropometrics = {
            'height': 1.75,  # Reference height in meters
            'weight': 75.0,  # Reference weight in kg
            'limb_lengths': {
                'arm': 0.73,  # Upper + lower arm
                'leg': 0.85,  # Upper + lower leg
                'torso': 0.52
            }
        }
        
    def set_reference_pattern(self, movement_type: str, joint_angles: Dict[str, np.ndarray]):
        """
        Set reference pattern for a specific movement type
        
        Args:
            movement_type: Type of movement (e.g., 'bowling', 'batting')
            joint_angles: Dictionary of joint angles time series data
        """
        self.reference_patterns[movement_type] = joint_angles
        
    def compare_with_reference(self, 
                             current_pattern: Dict[str, np.ndarray], 
                             movement_type: str,
                             player_height: float = None,
                             player_weight: float = None,
                             player_limb_lengths: Dict[str, float] = None) -> Tuple[float, Dict[str, float], Dict[str, List[float]]]:
        """
        Compare current movement pattern with the reference pattern
        
        Args:
            current_pattern: Dictionary of current joint angles
            movement_type: Type of movement to compare against
            player_height: Height of the player in meters
            player_weight: Weight of the player in kilograms
            player_limb_lengths: Dictionary of player's limb lengths
            
        Returns:
            overall_similarity: Overall similarity score (0-1)
            joint_similarities: Dictionary of similarity scores for each joint
            deviations: Dictionary of deviation time series for each joint
        """
        if movement_type not in self.reference_patterns:
            raise ValueError(f"No reference pattern found for {movement_type}")
            
        reference = self.reference_patterns[movement_type]
        joint_similarities = {}
        deviations = {}
        
        # Special handling for spin bowling
        is_spin = movement_type in ['off_spin', 'leg_spin']
        
        # Special handling for defensive shots
        is_defense = movement_type in ['front_foot_defense', 'back_foot_defense']
        
        # Normalize patterns based on anthropometric data if available
        normalized_current = current_pattern.copy()
        if player_height is not None:
            # Normalize displacement-based measurements by height
            for joint, data in normalized_current.items():
                normalized_current[joint] = normalize_by_height(
                    data, player_height, self.reference_anthropometrics['height']
                )
                
        if player_limb_lengths is not None:
            # Normalize by individual limb lengths
            normalized_current = normalize_by_limb_length(
                normalized_current, 
                player_limb_lengths,
                self.reference_anthropometrics['limb_lengths']
            )
            
        # Adjust comparison thresholds based on player's build
        if player_height is not None and player_weight is not None:
            bmi_factor = calculate_bmi_factor(player_weight, player_height)
            # Adjust similarity thresholds for larger/smaller players
            similarity_threshold_adjustment = 1.0 / bmi_factor
        
        for joint in current_pattern.keys():
            if joint in reference:
                # Calculate DTW distance
                distance, _ = fastdtw(current_pattern[joint], reference[joint])
                
                # Calculate correlation
                correlation, _ = pearsonr(current_pattern[joint], reference[joint])
                
                # Combine metrics into similarity score (0-1)
                max_distance = np.max([np.ptp(current_pattern[joint]), np.ptp(reference[joint])])
                distance_score = 1 - (distance / (max_distance * len(current_pattern[joint])))
                similarity = (distance_score + max(0, correlation)) / 2
                
                joint_similarities[joint] = similarity
                
                # Calculate deviations
                deviations[joint] = np.abs(current_pattern[joint] - reference[joint])
        
        # Calculate overall similarity score
        overall_similarity = np.mean(list(joint_similarities.values()))
        
        return overall_similarity, joint_similarities, deviations
    
    def detect_technique_errors(self, 
                              movement_data: Dict[str, np.ndarray], 
                              movement_type: str,
                              threshold: float = 0.15) -> Dict[str, List[Dict]]:
        """
        Detect technical errors in movement by comparing with reference patterns
        
        Args:
            movement_data: Dictionary of joint angles
            movement_type: Type of movement being analyzed
            threshold: Threshold for error detection
            
        Returns:
            Dictionary of detected errors for each joint
        """
        _, _, deviations = self.compare_with_reference(movement_data, movement_type)
        errors = {}
        
        for joint, deviation in deviations.items():
            joint_errors = []
            # Find peaks in deviation that exceed threshold
            peaks = np.where(deviation > threshold)[0]
            
            if len(peaks) > 0:
                # Group consecutive frames into single errors
                error_ranges = []
                start = peaks[0]
                
                for i in range(1, len(peaks)):
                    if peaks[i] - peaks[i-1] > 1:
                        error_ranges.append({
                            'start_frame': int(start),
                            'end_frame': int(peaks[i-1]),
                            'max_deviation': float(np.max(deviation[start:peaks[i-1]+1]))
                        })
                        start = peaks[i]
                
                # Add last error range
                error_ranges.append({
                    'start_frame': int(start),
                    'end_frame': int(peaks[-1]),
                    'max_deviation': float(np.max(deviation[start:peaks[-1]+1]))
                })
                
                errors[joint] = error_ranges
            
        return errors
    
    def assess_injury_risk(self, 
                          joint_angles: Dict[str, np.ndarray],
                          joint_velocities: Dict[str, np.ndarray],
                          movement_type: str = None,
                          player_height: float = None,
                          player_weight: float = None) -> Dict[str, Dict]:
        """
        Assess injury risk based on joint angles and velocities
        
        Args:
            joint_angles: Dictionary of joint angles over time
            joint_velocities: Dictionary of joint angular velocities
            movement_type: Type of movement being analyzed
            player_height: Height of the player in meters
            player_weight: Weight of the player in kilograms
            
        Returns:
            Dictionary containing risk assessments for each joint
        """
        risk_assessment = {}
        
        from ..preprocessing.anthropometric import calculate_exercise_rom_limits
        
        # Get exercise-specific ROM limits if available
        if movement_type and movement_type in ['push', 'pull', 'hinge', 'squat', 'lunge'] and player_height and player_weight:
            angle_thresholds = calculate_exercise_rom_limits(player_height, player_weight, movement_type)
        else:
            # Default thresholds
            angle_thresholds = {
                'shoulder': {'min': -90, 'max': 180},
                'elbow': {'min': 0, 'max': 145},
                'spine': {'min': -30, 'max': 30},
                'knee': {'min': 0, 'max': 145},
                'ankle': {'min': -20, 'max': 45}
            }
        
        # Define and adjust velocity thresholds
        base_velocity_thresholds = {
            'shoulder': 500,
            'elbow': 400,
            'spine': 200,
            'knee': 300,
            'ankle': 250
        }
        
        if player_height and player_weight:
            from ..preprocessing.anthropometric import adjust_thresholds_by_anthropometrics
            velocity_thresholds = adjust_thresholds_by_anthropometrics(
                base_velocity_thresholds,
                player_height,
                player_weight,
                movement_type
            )
        else:
            velocity_thresholds = base_velocity_thresholds
        
        for joint in joint_angles.keys():
            if joint in angle_thresholds:
                angles = joint_angles[joint]
                velocities = joint_velocities.get(joint, np.zeros_like(angles))
                
                # Check angle limits
                angle_violations = np.logical_or(
                    angles < angle_thresholds[joint]['min'],
                    angles > angle_thresholds[joint]['max']
                )
                
                # Check velocity limits
                velocity_violations = np.abs(velocities) > velocity_thresholds.get(joint, float('inf'))
                
                # Calculate risk metrics
                risk_assessment[joint] = {
                    'angle_violation_percentage': float(np.mean(angle_violations) * 100),
                    'velocity_violation_percentage': float(np.mean(velocity_violations) * 100),
                    'max_angle': float(np.max(angles)),
                    'min_angle': float(np.min(angles)),
                    'max_velocity': float(np.max(np.abs(velocities))),
                    'overall_risk_score': float((np.mean(angle_violations) + np.mean(velocity_violations)) * 50)
                }
        
        return risk_assessment
    
    def calculate_performance_metrics(self, 
                                   movement_data: Dict[str, np.ndarray],
                                   movement_type: str) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics including efficiency and symmetry
        
        Args:
            movement_data: Dictionary of joint angles and positions
            movement_type: Type of movement being analyzed
            
        Returns:
            Dictionary of performance metrics
        """
        # Compare with reference pattern
        overall_similarity, joint_similarities, _ = self.compare_with_reference(
            movement_data, movement_type
        )
        
        # Calculate symmetry scores
        symmetry_scores = calculate_symmetry_score(movement_data)
        
        # Calculate smoothness of movement
        smoothness_scores = {}
        for joint, angles in movement_data.items():
            # Calculate jerk (third derivative of position)
            jerk = np.diff(angles, n=3, axis=0)
            smoothness_scores[joint] = float(1 / (1 + np.mean(np.abs(jerk))))
            
        # Special handling for spin bowling
        if movement_type in ['off_spin', 'leg_spin'] and 'wrist' in movement_data:
            from ..feature_extraction.joint_features import wrist_flex_analysis
            wrist_metrics = wrist_flex_analysis(movement_data['wrist'], is_spin=True)
            spin_metrics = {
                'wrist_control': float(wrist_metrics['wrist_flex_rate']),
                'spin_potential': float(wrist_metrics['max_flex_rate']),
                'wrist_flexibility': float(wrist_metrics['wrist_flex_rom'])
            }
        else:
            spin_metrics = {}
            
        # Special handling for defensive shots
        if movement_type in ['front_foot_defense', 'back_foot_defense']:
            # Calculate head stability
            head_stability = 1.0 - float(np.std(movement_data.get('head', [0])))
            
            # Calculate balance metrics
            hip_angles = movement_data.get('hip', [0])
            knee_angles = movement_data.get('knee', [0])
            balance_score = float(1.0 - (np.std(hip_angles) + np.std(knee_angles)) / 2)
            
            defense_metrics = {
                'head_stability': head_stability,
                'balance_score': balance_score,
                'foot_movement_timing': float(overall_similarity)  # Using overall similarity as proxy
            }
        else:
            defense_metrics = {}
            
        # Special handling for gym exercises
        if movement_type in ['push', 'pull', 'hinge', 'squat', 'lunge']:
            # Calculate exercise-specific metrics
            if 'spine' in movement_data:
                spine_stability = 1.0 - float(np.std(movement_data['spine']))
            else:
                spine_stability = 1.0
                
            # Calculate tempo consistency
            joint_velocities = {}
            for joint, angles in movement_data.items():
                velocities = np.gradient(angles)
                joint_velocities[joint] = float(np.std(velocities))
            tempo_score = 1.0 - float(np.mean(list(joint_velocities.values())))
            
            # Calculate ROM utilization
            rom_scores = {}
            for joint, angles in movement_data.items():
                actual_rom = float(np.ptp(angles))
                expected_rom = 90.0  # Default expected ROM
                rom_scores[joint] = min(1.0, actual_rom / expected_rom)
            
            exercise_metrics = {
                'spine_stability': spine_stability,
                'tempo_consistency': tempo_score,
                'rom_utilization': float(np.mean(list(rom_scores.values()))),
                'movement_control': float(overall_similarity * (1 + spine_stability) / 2)
            }
        else:
            exercise_metrics = {}
        
        # Combine metrics
        metrics = {
            'overall_technique_similarity': float(overall_similarity),
            'overall_symmetry': float(np.mean(list(symmetry_scores.values()))),
            'movement_smoothness': float(np.mean(list(smoothness_scores.values()))),
            'technique_consistency': float(np.std(list(joint_similarities.values()))),
            'efficiency_score': float(
                (overall_similarity + np.mean(list(symmetry_scores.values())) +
                 np.mean(list(smoothness_scores.values()))) / 3
            )
        }
        
        # Add specialized metrics if available
        metrics.update(spin_metrics)
        metrics.update(defense_metrics)
        metrics.update(exercise_metrics)
        
        return metrics
