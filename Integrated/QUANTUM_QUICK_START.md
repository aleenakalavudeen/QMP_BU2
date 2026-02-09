# Quantum Modules Quick Start Guide

This guide provides a quick overview of how to use quantum modules to tackle YOLO bottlenecks in your traffic perception system.

## Overview

The quantum modules provide **quantum-inspired** optimizations (classical algorithms inspired by quantum mechanics) that can be used immediately without quantum hardware. These modules address key YOLO bottlenecks:

1. **Frame Processing Overhead** → Quantum Frame Selector
2. **NMS Bottleneck** → Quantum NMS
3. **Multi-Model Fusion** → Quantum Ensemble Fusion
4. **Hyperparameter Tuning** → Quantum Optimizer

## Quick Integration

### Option 1: Use the Enhanced Wrapper Class

```python
from quantum_integration_example import QuantumEnhancedTrafficPerception

# Create quantum-enhanced system
system = QuantumEnhancedTrafficPerception(use_quantum_modules=True)

# Process a video
stats = system.process_video('video.mp4', output_path='output.mp4')
print(f"Quantum savings: {stats['quantum_savings_pct']:.1f}%")
```

### Option 2: Integrate into Existing Code

Add to your `main.py`:

```python
from quantum_modules import QuantumFrameSelector, QuantumNMS, QuantumEnsembleFusion

class IntegratedTrafficPerception:
    def __init__(self, ...):
        # ... existing code ...
        
        # Add quantum modules
        self.quantum_frame_selector = QuantumFrameSelector(importance_threshold=0.5)
        self.quantum_nms = QuantumNMS(iou_threshold=0.45)
        self.quantum_fusion = QuantumEnsembleFusion(n_models=4)
    
    def process_frame(self, frame, ...):
        # Step 1: Quantum frame selection
        if not self.quantum_frame_selector.should_process(frame):
            return {'detections': [], 'frame_skipped': True}
        
        # Step 2: Process with existing models
        results = self._process_with_models(frame, ...)
        
        # Step 3: Apply quantum NMS
        for model_name in ['sign', 'signal', 'anomaly']:
            if model_name in results:
                results[model_name] = self._apply_quantum_nms(
                    results[model_name], self.quantum_nms
                )
        
        # Step 4: Quantum fusion (if multiple models)
        if len(self.selected_models) > 1:
            detections_list = [results.get(m, []) for m in self.selected_models]
            results['fused'] = self.quantum_fusion.fuse_detections(detections_list)
        
        return results
```

## Module Usage Examples

### 1. Quantum Frame Selector

Skip unimportant frames to reduce processing by 30-50%:

```python
from quantum_modules import QuantumFrameSelector

selector = QuantumFrameSelector(importance_threshold=0.5)

for frame in video_frames:
    if selector.should_process(frame):
        # Process frame
        results = process_frame(frame)
    else:
        # Skip frame (use previous results or skip)
        continue
```

### 2. Quantum NMS

Faster NMS for large detection sets:

```python
from quantum_modules import QuantumNMS
import numpy as np

quantum_nms = QuantumNMS(iou_threshold=0.45)

# Your detections
boxes = np.array([[x1, y1, x2, y2], ...])  # [N, 4]
scores = np.array([0.9, 0.8, 0.7, ...])     # [N]
classes = np.array([0, 1, 0, ...])         # [N]

# Apply quantum NMS
keep_indices = quantum_nms.suppress(boxes, scores, classes)

# Filter detections
filtered_detections = [detections[i] for i in keep_indices]
```

### 3. Quantum Ensemble Fusion

Combine detections from multiple models:

```python
from quantum_modules import QuantumEnsembleFusion

fusion = QuantumEnsembleFusion(n_models=4, fusion_method='quantum_voting')

# Detections from different models
sign_detections = [...]      # From sign model
signal_detections = [...]    # From signal model
anomaly_detections = [...]   # From anomaly model
risk_detections = [...]      # From risk model

# Fuse using quantum voting
fused = fusion.fuse_detections([
    sign_detections,
    signal_detections,
    anomaly_detections,
    risk_detections
])
```

### 4. Quantum Optimizer

Optimize confidence thresholds:

```python
from quantum_modules import QuantumInspiredOptimizer

optimizer = QuantumInspiredOptimizer(n_particles=20, max_iterations=50)

# Collect detection history
detection_history = [
    {'scores': [0.9, 0.8, 0.7, ...]},
    {'scores': [0.85, 0.75, 0.65, ...]},
    # ... more detection results
]

# Optimize threshold
optimal_threshold = optimizer.optimize_confidence_threshold(
    detection_history,
    target_precision=0.8,
    target_recall=0.7
)

print(f"Optimal confidence threshold: {optimal_threshold:.3f}")
```

## Configuration

Create a config file (`quantum_config.json`):

```json
{
    "n_particles": 20,
    "max_iterations": 50,
    "importance_threshold": 0.5,
    "history_size": 10,
    "iou_threshold": 0.45,
    "use_quantum_nms": true,
    "n_models": 4,
    "fusion_method": "quantum_voting"
}
```

Load and use:

```python
import json
from quantum_modules import create_quantum_modules

with open('quantum_config.json', 'r') as f:
    config = json.load(f)

modules = create_quantum_modules(config)
```

## Expected Performance Improvements

| Module | Bottleneck | Expected Improvement |
|--------|-----------|---------------------|
| Frame Selector | Video processing | 30-50% fewer frames |
| Quantum NMS | NMS processing | 2-4x faster (large sets) |
| Ensemble Fusion | Multi-model fusion | 1.5-2x faster |
| Optimizer | Hyperparameter tuning | 2-3x faster search |

## Command Line Usage

```bash
# Process video with quantum modules
python quantum_integration_example.py --source video.mp4 --use-quantum --output output.mp4

# Process webcam with quantum modules
python quantum_integration_example.py --source webcam --use-quantum

# Use custom config
python quantum_integration_example.py --source video.mp4 --use-quantum --config quantum_config.json
```

## Troubleshooting

### Module Not Found
```bash
# Ensure quantum_modules.py is in the same directory
# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/Integrated"
```

### Performance Not Improved
- **Frame Selector**: Lower `importance_threshold` (e.g., 0.3) to skip more frames
- **Quantum NMS**: Only beneficial for >50 detections per frame
- **Ensemble Fusion**: Only useful when multiple models are active

### Memory Issues
- Reduce `n_particles` in optimizer
- Reduce `history_size` in frame selector
- Disable quantum NMS for small detection sets

## Next Steps

1. **Start Simple**: Enable frame selector first (biggest impact)
2. **Measure**: Compare processing times with/without quantum modules
3. **Tune**: Adjust thresholds based on your use case
4. **Scale**: Enable all modules once comfortable

## Advanced: True Quantum Computing

For true quantum acceleration (requires quantum hardware):

1. Install quantum libraries:
   ```bash
   pip install qiskit pennylane cirq
   ```

2. Use quantum simulators or connect to quantum hardware (IBM Quantum, Google Quantum AI)

3. See `QUANTUM_MODULE_APPLICATIONS.md` for advanced quantum implementations

## References

- Full documentation: `QUANTUM_MODULE_APPLICATIONS.md`
- Implementation: `quantum_modules.py`
- Integration example: `quantum_integration_example.py`
