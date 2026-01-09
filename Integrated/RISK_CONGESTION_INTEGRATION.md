# Risk & Congestion Detection Integration

## Overview

The Driving Risk Assessment and Congestion Detection model has been successfully integrated into the Integrated Traffic Perception System. The integration follows a strict non-intrusive approach, preserving the original model's behavior exactly as-is.

## Integration Architecture

### Wrapper Module: `risk_congestion_module.py`

A wrapper module (`risk_congestion_module.py`) was created that:
- Imports from the original `driving_risk_and_congestion/src` directory without modification
- Provides a `RiskCongestionDetector` class with a standard `detect()` interface
- Preserves all original risk computation logic, PCU calculations, and constants
- Maintains independent configuration and preprocessing

### Integration Points in `main.py`

1. **Model Registration**: Added `'risk_congestion'` to `VALID_MODELS`
2. **Initialization**: Added risk_congestion detector initialization in `__init__()`
3. **Thread-Safe Detection**: Added thread-safe detection methods for parallel processing
4. **Frame Processing**: Integrated into `process_frame()` method
5. **Result Tracking**: Added tracking for risk_congestion detections and metrics
6. **Output Formatting**: Added risk/congestion metrics to results dictionary

## Key Features

### Preserved Original Behavior

- ✅ All risk computation logic (areas, x_disps, combined metrics, B_sum)
- ✅ Risk label classification (LOW, MODERATE, HIGH, CRITICAL) using original centroids
- ✅ Risk label smoothing with exponential decay (for video)
- ✅ PCU computation with original filtering logic (ROI, distance proxy)
- ✅ Original preprocessing (crop bottom 65%, normalization)
- ✅ Original model architecture (SqueezeDet)
- ✅ Original detection filtering (NMS, score thresholding)

### Integration Features

- ✅ Standardized detection format compatible with other models
- ✅ Parallel processing support (thread-safe)
- ✅ Risk and congestion metrics included in results
- ✅ Frame-by-frame processing for videos
- ✅ Visualization with bounding boxes (orange color)
- ✅ Performance timing tracking

## Usage

### Basic Usage

```python
from main import IntegratedTrafficPerception

# Initialize with risk_congestion enabled
pipeline = IntegratedTrafficPerception(
    selected_models=['sign', 'signal', 'anomaly', 'risk_congestion'],
    device='cuda',  # or 'cpu'
    use_parallel=True
)

# Process a frame
results = pipeline.process_frame(frame, frame_id=1)

# Access results
detections = results['detections']  # All detections including vehicles/pedestrians
risk_metrics = results['risk_metrics']  # Risk computation results
congestion_metrics = results['congestion_metrics']  # PCU computation results
```

### Risk Metrics Structure

```python
risk_metrics = {
    'counts': [c0, c1, c2, c3, c4],  # Counts per class
    'areas': [a0, a1, a2, a3, a4],   # Normalized areas per class
    'x_disps': [d0, d1, d2, d3, d4], # X-axis displacements per class
    'combined': [b0, b1, b2, b3, b4], # Combined metrics per class
    'B_sum': float,                  # Total combined metric
    'risk_label': str,                # Raw risk label: LOW/MODERATE/HIGH/CRITICAL
    'smoothed_label': str | None     # Smoothed label (after window warm-up)
}
```

### Congestion Metrics Structure

```python
congestion_metrics = {
    'counts': [pc_c0, pc_c1, pc_c2, pc_c3, pc_c4],  # Filtered counts per class
    'pcu_per_class': [pc_p0, pc_p1, pc_p2, pc_p3, pc_p4],  # PCU per class
    'pcu_sum': float  # Total PCU sum
}
```

## File Structure

```
Integrated/
├── risk_congestion_module.py    # Wrapper module (NEW)
├── main.py                       # Updated with integration
└── ...

driving_risk_and_congestion/
├── src/                          # Original model (UNMODIFIED)
│   ├── demoidd_video_riskest_cong.py
│   ├── demoidd_riskest_cong.py
│   ├── model/
│   ├── engine/
│   ├── datasets/
│   └── utils/
└── models/                       # Model weights
```

## Configuration

The risk_congestion model uses its own configuration system:
- Model weights: `../driving_risk_and_congestion/models/model_best_cropped_full_AdamW.pth`
- Input size: (704, 1920) for cropped HD (bottom 65%)
- Classes: lmv, person, two_wheeler, three_wheeler, hmv
- Risk centroids: Pre-computed values (unchanged)
- PCU factors: [1.00, 0.00, 0.75, 2.00, 3.70]

## Output Compatibility

The integrated system produces outputs that are **bit-for-bit compatible** with standalone execution:
- Same detection boxes (after coordinate transformation)
- Same risk labels (LOW/MODERATE/HIGH/CRITICAL)
- Same PCU values
- Same frame-by-frame behavior

## Notes

1. **No Shared State**: The risk_congestion model maintains its own:
   - Configuration objects
   - Model weights
   - Preprocessing functions
   - Postprocessing logic

2. **Independent Execution**: The model runs independently of other detection models, ensuring no interference.

3. **Video Processing**: For video processing, frame_id should be passed to enable proper risk label smoothing.

4. **Performance**: The model can run in parallel with other models using thread-safe detection methods.

## Testing

To verify the integration works correctly:

1. Run standalone: `python driving_risk_and_congestion/src/demoidd_video_riskest_cong.py`
2. Run integrated: `python Integrated/main.py --models risk_congestion --input video.mp4`
3. Compare outputs: Risk labels, PCU values, and detection boxes should match

## Dependencies

The integration requires:
- Original `driving_risk_and_congestion/src` directory structure intact
- Model weights file at expected path
- All original dependencies (torch, numpy, cv2, etc.)

