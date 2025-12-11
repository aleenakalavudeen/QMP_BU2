# Quick Start Guide

## Running the Integrated Pipeline

### Important: Paths with Spaces Must Be Quoted

If your paths contain spaces (like `"test image"`), you **must** put quotes around them.

### Examples

**Process an image:**
```bash
# Windows PowerShell or CMD
python main.py --source "C:\Quantum enhanced Traffic Perception model\Integrated\test image\original.jpg" --output "C:\Quantum enhanced Traffic Perception model\Integrated\output\output.jpg"

# Or use relative paths (easier!)
python main.py --source "test image\original.jpg" --output "output\output.jpg"
```

**Process a video:**
```bash
python main.py --source "path\to\video.mp4" --output "output\annotated_video.mp4"
```

**Process webcam:**
```bash
python main.py --source webcam
```

**Process without displaying (save only):**
```bash
python main.py --source "image.jpg" --output "output.jpg" --no-show
```

### Common Issues

**Error: "unrecognized arguments"**
- **Cause**: Paths with spaces not quoted
- **Fix**: Put quotes around paths: `--source "path with spaces\file.jpg"`

**Error: "Source file not found"**
- **Cause**: Wrong path or file doesn't exist
- **Fix**: Check the path exists, use relative paths, or use forward slashes: `test image/original.jpg`

### Tips

1. **Use relative paths** - Easier and less error-prone:
   ```bash
   python main.py --source "test image\original.jpg"
   ```

2. **Create output directory first**:
   ```bash
   mkdir output
   python main.py --source "test image\original.jpg" --output "output\result.jpg"
   ```

3. **Drag and drop in PowerShell** - Type the command, then drag the file into the terminal to auto-quote:
   ```bash
   python main.py --source [drag file here]
   ```

