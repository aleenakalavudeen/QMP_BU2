"""
Simple test script to verify all three wrapper modules work correctly.

This script tests loading each module and running a basic detection.
Run this to make sure everything is set up correctly before proceeding.
"""

import cv2
import numpy as np
from pathlib import Path

def test_sign_module():
    """Test the sign detection module."""
    print("\n" + "="*60)
    print("Testing Sign Detection Module")
    print("="*60)
    
    try:
        from sign_module import SignDetector
        
        print("\n1. Initializing sign detector...")
        detector = SignDetector()
        print("   ✓ Sign detector initialized!")
        
        # Create a dummy test image (black image)
        print("\n2. Creating test image...")
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        print("   ✓ Test image created (640x640 black image)")
        
        print("\n3. Running detection...")
        detections = detector.detect(test_image)
        print(f"   ✓ Detection completed! Found {len(detections)} signs")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_signal_module():
    """Test the signal detection module."""
    print("\n" + "="*60)
    print("Testing Signal Detection Module")
    print("="*60)
    
    try:
        from signal_module import SignalDetector
        
        print("\n1. Initializing signal detector...")
        detector = SignalDetector()
        print("   ✓ Signal detector initialized!")
        
        # Create a dummy test image
        print("\n2. Creating test image...")
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        print("   ✓ Test image created (640x640 black image)")
        
        print("\n3. Running detection...")
        detections = detector.detect(test_image)
        print(f"   ✓ Detection completed! Found {len(detections)} signals")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_road_module():
    """Test the road anomaly detection module."""
    print("\n" + "="*60)
    print("Testing Road Anomaly Detection Module")
    print("="*60)
    
    try:
        from road_module import RoadAnomalyDetector
        
        print("\n1. Initializing road anomaly detector...")
        detector = RoadAnomalyDetector()
        print("   ✓ Road anomaly detector initialized!")
        
        # Create a dummy test image
        print("\n2. Creating test image...")
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        print("   ✓ Test image created (640x640 black image)")
        
        print("\n3. Running detection...")
        detections = detector.detect(test_image)
        print(f"   ✓ Detection completed! Found {len(detections)} anomalies")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

if __name__ == "__main__":
    """
    Run all tests to verify the wrapper modules work correctly.
    """
    print("\n" + "="*60)
    print("WRAPPER MODULES TEST SUITE")
    print("="*60)
    print("\nThis script tests if all three wrapper modules can be loaded")
    print("and run basic detection. It uses dummy black images for testing.")
    print("\nNote: This only tests if the modules load correctly.")
    print("      For real detection, use actual traffic images.")
    
    results = []
    
    # Test each module
    results.append(("Sign Detection", test_sign_module()))
    results.append(("Signal Detection", test_signal_module()))
    results.append(("Road Anomaly Detection", test_road_module()))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:30s} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("All modules loaded successfully! ✓")
        print("\nNext steps:")
        print("1. Test with real images to verify detections")
        print("2. Proceed to Step 2: Create integrated pipeline")
    else:
        print("Some modules failed to load. Please check:")
        print("1. Weight files exist in correct locations")
        print("2. Required dependencies are installed")
        print("3. YOLOv5/YOLOv8 repositories are available")
    print("="*60 + "\n")

