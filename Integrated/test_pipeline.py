"""
Simple test script for the integrated pipeline.

This script tests the main pipeline with a dummy image to verify
everything works together correctly.
"""

import cv2
import numpy as np
from main import IntegratedTrafficPerception

def test_pipeline():
    """Test the integrated pipeline with a dummy image."""
    print("=" * 70)
    print("Testing Integrated Pipeline")
    print("=" * 70)
    
    try:
        # Initialize pipeline
        print("\n1. Initializing pipeline...")
        pipeline = IntegratedTrafficPerception()
        print("   ✓ Pipeline initialized!")
        
        # Create a dummy test image (640x480, random colors)
        print("\n2. Creating test image...")
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print("   ✓ Test image created (640x480)")
        
        # Process frame
        print("\n3. Processing frame through all models...")
        results = pipeline.process_frame(test_image)
        print("   ✓ Frame processed!")
        
        # Check output structure
        print("\n4. Verifying output structure...")
        
        required_keys = ['detections', 'confidence_summaries', 'fused_feature_vector', 'annotated_frame']
        for key in required_keys:
            if key in results:
                print(f"   ✓ '{key}' present")
            else:
                print(f"   ✗ '{key}' missing!")
                return False
        
        # Check detections format
        print("\n5. Checking detection format...")
        if len(results['detections']) >= 0:  # Can be empty for dummy image
            sample_det = results['detections'][0] if results['detections'] else None
            if sample_det:
                det_keys = ['model_type', 'class_name', 'confidence', 'bbox', 'feature_vector']
                for key in det_keys:
                    if key in sample_det:
                        print(f"   ✓ Detection has '{key}'")
                    else:
                        print(f"   ✗ Detection missing '{key}'!")
                        return False
        
        # Check confidence summaries
        print("\n6. Checking confidence summaries...")
        conf_keys = ['sign_model', 'signal_model', 'anomaly_model']
        for key in conf_keys:
            if key in results['confidence_summaries']:
                print(f"   ✓ '{key}' present")
            else:
                print(f"   ✗ '{key}' missing!")
                return False
        
        # Check fused feature vector
        print("\n7. Checking fused feature vector...")
        fused = results['fused_feature_vector']
        if isinstance(fused, np.ndarray) and len(fused) == 256:
            print(f"   ✓ Fused vector: shape={fused.shape}, dtype={fused.dtype}")
        else:
            print(f"   ✗ Fused vector incorrect: shape={fused.shape if isinstance(fused, np.ndarray) else type(fused)}")
            return False
        
        # Check annotated frame
        print("\n8. Checking annotated frame...")
        annotated = results['annotated_frame']
        if isinstance(annotated, np.ndarray) and annotated.shape == test_image.shape:
            print(f"   ✓ Annotated frame: shape={annotated.shape}")
        else:
            print(f"   ✗ Annotated frame incorrect!")
            return False
        
        print("\n" + "=" * 70)
        print("All tests passed! ✓")
        print("=" * 70)
        print("\nPipeline is ready to use!")
        print("\nNext steps:")
        print("1. Test with real images: python main.py --source path/to/image.jpg")
        print("2. Test with video: python main.py --source path/to/video.mp4")
        print("3. Test with webcam: python main.py --source webcam")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_pipeline()

