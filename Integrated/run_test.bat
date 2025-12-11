@echo off
REM Simple batch script to run the pipeline with your test image
REM This avoids quote/space issues in PowerShell/CMD

python main.py --source "test image\original.jpg" --output "output\output.jpg"

pause

