"""
Feature Extraction and Fusion Module

This module handles:
1. Extracting feature vectors from detections
2. Pooling features from multiple detections per model
3. Fusing features from all three models into a single compact vector
4. Preparing features for quantum module input

Author: Integrated Traffic Perception System
"""

import numpy as np
from typing import List, Dict, Tuple


def pool_features(feature_vectors: List[np.ndarray], method: str = 'mean') -> np.ndarray:
    """
    Pool multiple feature vectors into a single vector.
    
    When a model detects multiple objects, we need to combine their features
    into one representative vector. This function does that pooling.
    
    Args:
        feature_vectors (list): List of feature vectors (each is a numpy array)
        method (str): Pooling method - 'mean', 'max', or 'concat'
                     'mean': Average all vectors (recommended)
                     'max': Take element-wise maximum
                     'concat': Concatenate all vectors (only if few detections)
    
    Returns:
        numpy.ndarray: Single pooled feature vector
    """
    if len(feature_vectors) == 0:
        # Return zero vector if no detections
        return np.zeros(128, dtype=np.float32)
    
    if len(feature_vectors) == 1:
        # Return the single vector as-is
        return feature_vectors[0].astype(np.float32)
    
    # Stack all vectors into a matrix
    stacked = np.stack(feature_vectors, axis=0)
    
    if method == 'mean':
        # Average pooling: take mean across all detections
        pooled = np.mean(stacked, axis=0)
    elif method == 'max':
        # Max pooling: take maximum value for each dimension
        pooled = np.max(stacked, axis=0)
    elif method == 'concat':
        # Concatenation: combine all vectors (use with caution - can get large!)
        pooled = stacked.flatten()
        # Limit size to prevent explosion
        if len(pooled) > 512:
            pooled = pooled[:512]
    else:
        raise ValueError(f"Unknown pooling method: {method}. Use 'mean', 'max', or 'concat'")
    
    return pooled.astype(np.float32)


def extract_model_features(detections: List[Dict]) -> Tuple[np.ndarray, List[float]]:
    """
    Extract pooled features and confidence scores from a list of detections.
    
    This function takes all detections from one model and:
    1. Extracts all feature vectors
    2. Pools them into a single vector
    3. Collects all confidence scores
    
    Args:
        detections (list): List of detection dictionaries from one model
    
    Returns:
        tuple: (pooled_feature_vector, confidence_scores_list)
               - pooled_feature_vector: Single feature vector representing all detections
               - confidence_scores_list: List of all confidence scores
    """
    if len(detections) == 0:
        # Return zero vector and empty confidence list
        return np.zeros(128, dtype=np.float32), []
    
    # Extract all feature vectors
    feature_vectors = [det['feature_vector'] for det in detections]
    
    # Extract all confidence scores
    confidence_scores = [det['confidence'] for det in detections]
    
    # Pool the features (using mean pooling by default)
    pooled_features = pool_features(feature_vectors, method='mean')
    
    return pooled_features, confidence_scores


def fuse_features(
    sign_features: np.ndarray,
    signal_features: np.ndarray,
    anomaly_features: np.ndarray,
    sign_confidences: List[float],
    signal_confidences: List[float],
    anomaly_confidences: List[float],
    include_confidence: bool = True,
    target_size: int = 256
) -> np.ndarray:
    """
    Fuse features from all three models into a single compact vector.
    
    This is a CRITICAL function for quantum module integration. It combines:
    - Pooled features from sign model
    - Pooled features from signal model
    - Pooled features from anomaly model
    - Optional aggregated confidence scores
    
    The output is a fixed-length vector suitable for quantum processing.
    
    Args:
        sign_features (np.ndarray): Pooled features from sign model
        signal_features (np.ndarray): Pooled features from signal model
        anomaly_features (np.ndarray): Pooled features from anomaly model
        sign_confidences (list): Confidence scores from sign model
        signal_confidences (list): Confidence scores from signal model
        anomaly_confidences (list): Confidence scores from anomaly model
        include_confidence (bool): Whether to include confidence statistics
        target_size (int): Target size for final fused vector (default: 256)
    
    Returns:
        numpy.ndarray: Fused feature vector of fixed length (target_size)
    """
    # Step 1: Concatenate all three feature vectors
    # Each is 128 dimensions, so concatenated = 384 dimensions
    concatenated = np.concatenate([
        sign_features,
        signal_features,
        anomaly_features
    ])
    
    # Step 2: Add confidence statistics if requested
    if include_confidence:
        # Calculate aggregated confidence statistics
        sign_mean_conf = np.mean(sign_confidences) if sign_confidences else 0.0
        signal_mean_conf = np.mean(signal_confidences) if signal_confidences else 0.0
        anomaly_mean_conf = np.mean(anomaly_confidences) if anomaly_confidences else 0.0
        
        sign_max_conf = np.max(sign_confidences) if sign_confidences else 0.0
        signal_max_conf = np.max(signal_confidences) if signal_confidences else 0.0
        anomaly_max_conf = np.max(anomaly_confidences) if anomaly_confidences else 0.0
        
        # Add confidence features (6 values: mean and max for each model)
        confidence_features = np.array([
            sign_mean_conf, sign_max_conf,
            signal_mean_conf, signal_max_conf,
            anomaly_mean_conf, anomaly_max_conf
        ], dtype=np.float32)
        
        # Concatenate with feature vector
        concatenated = np.concatenate([concatenated, confidence_features])
        # Now we have 384 + 6 = 390 dimensions
    
    # Step 3: Reduce to target size using simple projection
    # We'll use a simple linear projection (can be replaced with PCA later)
    current_size = len(concatenated)
    
    if current_size <= target_size:
        # If already smaller or equal, pad with zeros
        fused = np.pad(concatenated, (0, target_size - current_size), mode='constant')
    else:
        # If larger, use simple downsampling/interpolation
        # Method 1: Take every nth element (simple but effective)
        step = current_size / target_size
        indices = [int(i * step) for i in range(target_size)]
        fused = concatenated[indices]
        
        # Alternative: Could use PCA here for better compression
        # For now, simple downsampling is sufficient
    
    # Step 4: Normalize to [0, 1] range for better quantum processing
    # This helps with quantum encoding later
    fused_min = np.min(fused)
    fused_max = np.max(fused)
    if fused_max > fused_min:
        fused = (fused - fused_min) / (fused_max - fused_min)
    else:
        fused = np.zeros_like(fused)
    
    return fused.astype(np.float32)


def get_feature_summary(fused_vector: np.ndarray) -> Dict:
    """
    Get a summary of the fused feature vector for debugging/inspection.
    
    Args:
        fused_vector (np.ndarray): Fused feature vector
    
    Returns:
        dict: Summary statistics
    """
    return {
        'size': len(fused_vector),
        'mean': float(np.mean(fused_vector)),
        'std': float(np.std(fused_vector)),
        'min': float(np.min(fused_vector)),
        'max': float(np.max(fused_vector)),
        'non_zero_count': int(np.count_nonzero(fused_vector))
    }


# Example usage and testing
if __name__ == "__main__":
    """
    Test the feature fusion functions.
    """
    print("Testing Feature Fusion Module")
    print("=" * 60)
    
    # Create dummy feature vectors (128 dimensions each)
    sign_features = np.random.rand(128).astype(np.float32)
    signal_features = np.random.rand(128).astype(np.float32)
    anomaly_features = np.random.rand(128).astype(np.float32)
    
    # Create dummy confidence scores
    sign_confidences = [0.85, 0.92, 0.78]
    signal_confidences = [0.95]
    anomaly_confidences = [0.67, 0.71]
    
    print("\n1. Testing feature pooling...")
    test_features = [sign_features, signal_features]
    pooled = pool_features(test_features, method='mean')
    print(f"   ✓ Pooled {len(test_features)} features into vector of size {len(pooled)}")
    
    print("\n2. Testing feature fusion...")
    fused = fuse_features(
        sign_features, signal_features, anomaly_features,
        sign_confidences, signal_confidences, anomaly_confidences,
        include_confidence=True,
        target_size=256
    )
    print(f"   ✓ Fused features into vector of size {len(fused)}")
    
    print("\n3. Feature summary:")
    summary = get_feature_summary(fused)
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Feature fusion module test completed! ✓")

