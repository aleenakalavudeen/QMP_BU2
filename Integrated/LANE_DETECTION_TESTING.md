# Testing Lane Detection from main.py

You can now test the lane detection model on custom datasets directly from `main.py` using the `--test-lane-detection` flag.

## Quick Start

```bash
cd Integrated
python main.py --test-lane-detection \
    --lane-images-dir "path/to/images" \
    --lane-labels-dir "path/to/labels" \
    --lane-output-dir "path/to/output"
```

## Command Line Options

### Required Arguments
- `--test-lane-detection`: Enable lane detection testing mode
- `--lane-images-dir`: Directory containing test images

### Optional Arguments
- `--lane-labels-dir`: Directory containing label files (if not provided, testing proceeds without ground truth)
- `--lane-output-dir`: Directory to save prediction masks and visualizations
- `--lane-label-format`: Label format - `auto` (detect automatically), `culanes`, `json`, or `mask` (default: `auto`)
- `--conf-threshold`: Confidence threshold for detections (default: 0.4)
- `--device`: Device to run on - `cpu`, `cuda/gpu`, or `auto` (default: `auto`)

## Examples

### Basic Usage (with ground truth)
```bash
python main.py --test-lane-detection \
    --lane-images-dir "C:\MyDataset\images" \
    --lane-labels-dir "C:\MyDataset\labels" \
    --lane-output-dir "C:\MyDataset\results"
```

### Without Ground Truth (inference only)
```bash
python main.py --test-lane-detection \
    --lane-images-dir "C:\MyDataset\images" \
    --lane-output-dir "C:\MyDataset\results"
```

### With Custom Confidence Threshold
```bash
python main.py --test-lane-detection \
    --lane-images-dir "C:\MyDataset\images" \
    --lane-labels-dir "C:\MyDataset\labels" \
    --conf-threshold 0.5
```

### With JSON Label Format
```bash
python main.py --test-lane-detection \
    --lane-images-dir "C:\MyDataset\images" \
    --lane-labels-dir "C:\MyDataset\labels" \
    --lane-label-format json
```

### Using GPU
```bash
python main.py --test-lane-detection \
    --lane-images-dir "C:\MyDataset\images" \
    --lane-labels-dir "C:\MyDataset\labels" \
    --device cuda
```

## Supported Label Formats

### 1. CuLanes Format (.txt)
Each line contains space-separated x y coordinates for one lane.

**Example:**
```
100 200 150 250 200 300 250 350
50 180 100 230 150 280
```

### 2. JSON Format (.json)
JSON file with lane annotations.

**Example:**
```json
{
  "lanes": [
    [[100, 200], [150, 250], [200, 300]],
    [[50, 180], [100, 230], [150, 280]]
  ]
}
```

### 3. Binary Mask Format (.png, .jpg)
Direct mask images where:
- **White pixels** = lanes
- **Black pixels** = background

## Output

The script generates:

1. **Evaluation Metrics** (printed to console):
   - IoU (Intersection over Union)
   - Precision
   - Recall
   - Dice Coefficient (F1)
   - Accuracy

2. **Output Files** (if `--lane-output-dir` specified):
   - `*_pred.png`: Binary prediction masks
   - `*_vis.jpg`: Visualization overlays
     - Green = predictions
     - Blue = ground truth (if available)

## Label File Naming

The script automatically tries to find label files using these naming conventions:

1. Same name as image: `image.jpg` → `image.txt`
2. Full image name: `image.jpg` → `image.jpg.txt`
3. CuLanes format: `image.jpg` → `image.lines.txt`
4. With suffix: `image.jpg` → `image_label.txt` or `image_gt.txt`

The script searches recursively in subdirectories, so your folder structure can be flexible.

## Comparison with test_culanes.py

Both methods work similarly, but using `main.py` has these advantages:
- ✅ Integrated with the main pipeline
- ✅ Uses the same lane detector instance
- ✅ Consistent with other testing modes
- ✅ Can leverage other pipeline features

The standalone `test_culanes.py` script is still available if you prefer a separate tool.

## Troubleshooting

### Issue: "Lane detection model not loaded"
- Make sure you're not using `--models` to exclude the lane model
- The lane model is automatically loaded when using `--test-lane-detection`

### Issue: "No image files found"
- Check that `--lane-images-dir` points to the correct directory
- Ensure images have extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`

### Issue: "Labels missing" in results
- Check that label files exist in `--lane-labels-dir`
- Verify label file names match image names
- Try specifying `--lane-label-format` explicitly instead of `auto`


