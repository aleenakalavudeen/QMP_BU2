# Quick Start Guide for CMD (Command Prompt)

## Running the Integrated Pipeline in CMD

### Solution 1: Use the Batch Script (Easiest)

Simply double-click `run_test.bat` or run:
```cmd
run_test.bat
```

### Solution 2: Use Relative Paths with Quotes

Make sure you're in the `Integrated` folder, then:

```cmd
python main.py --source "test image\original.jpg" --output "output\output.jpg"
```

### Solution 3: Use Forward Slashes

CMD accepts forward slashes too:

```cmd
python main.py --source "test image/original.jpg" --output "output/output.jpg"
```

### Solution 4: Use Full Paths (Properly Quoted)

```cmd
python main.py --source "C:\Quantum enhanced Traffic Perception model\Integrated\test image\original.jpg" --output "C:\Quantum enhanced Traffic Perception model\Integrated\output\output.jpg"
```

### Common CMD Issues

**Issue: "unrecognized arguments"**
- Make sure there are NO spaces before or after the `=` sign
- Wrong: `--source = "path"`
- Right: `--source "path"`

**Issue: Path not found**
- Make sure you're in the correct directory:
  ```cmd
  cd "C:\Quantum enhanced Traffic Perception model\Integrated"
  ```

**Issue: Output directory doesn't exist**
- Create it first:
  ```cmd
  mkdir output
  ```

### Step-by-Step for Your Case

1. Open CMD
2. Navigate to the Integrated folder:
   ```cmd
   cd "C:\Quantum enhanced Traffic Perception model\Integrated"
   ```
3. Create output folder (if it doesn't exist):
   ```cmd
   mkdir output
   ```
4. Run the command:
   ```cmd
   python main.py --source "test image\original.jpg" --output "output\output.jpg"
   ```

### Alternative: Copy-Paste This Exact Command

```cmd
cd /d "C:\Quantum enhanced Traffic Perception model\Integrated" && mkdir output 2>nul && python main.py --source "test image\original.jpg" --output "output\output.jpg"
```

The `cd /d` ensures it changes to the correct drive, and `mkdir output 2>nul` creates the folder silently if it already exists.

