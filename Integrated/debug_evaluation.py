"""
Debug script to understand why lane detection evaluation shows 0 TP, 70 FP, 0 FN

This script helps diagnose evaluation issues by:
1. Checking if ground truth labels are being loaded
2. Checking model_type matching
3. Checking IoU calculations
4. Showing sample detections and ground truth
"""

import cv2
import numpy as np
from pathlib import Path
import sys
from main import IntegratedTrafficPerception


def debug_single_image(image_path, label_path=None, model='lane'):
    """
    Debug evaluation for a single image.
    
    Args:
        image_path: Path to image
        label_path: Path to ground truth label (optional)
        model: Model to use ('lane' or list of models)
    """
    print("="*70)
    print("DEBUGGING LANE DETECTION EVALUATION")
    print("="*70)
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return
    
    h, w = image.shape[:2]
    print(f"\nImage: {image_path}")
    print(f"Size: {w}x{h}")
    
    # Initialize pipeline
    print(f"\nLoading model: {model}")
    pipeline = IntegratedTrafficPerception(selected_models=[model] if isinstance(model, str) else model)
    
    # Run detection
    print("\nRunning detection...")
    results = pipeline.process_frame(image)
    detections = results['detections']
    
    print(f"\nDetections found: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"  Detection {i+1}:")
        print(f"    model_type: {det['model_type']}")
        print(f"    class_name: {det['class_name']}")
        print(f"    confidence: {det['confidence']:.4f}")
        print(f"    bbox: {det['bbox']}")
        if 'mask' in det:
            mask = det['mask']
            if mask is not None:
                mask_area = np.sum(mask > 0.5)
                print(f"    mask_area: {mask_area} pixels")
    
    # Load ground truth if available
    if label_path and Path(label_path).exists():
        print(f"\nLoading ground truth from: {label_path}")
        gt_boxes = pipeline._load_yolo_labels(Path(label_path), w, h)
        print(f"Ground truth boxes found: {len(gt_boxes)}")
        
        for i, gt in enumerate(gt_boxes):
            print(f"  GT Box {i+1}:")
            print(f"    class_id: {gt.get('class_id', 'N/A')}")
            print(f"    class_name: {gt.get('class_name', 'N/A')}")
            print(f"    bbox: {gt.get('bbox', 'N/A')}")
            
            # Infer model type
            inferred_type = pipeline._infer_model_type_from_gt(gt)
            print(f"    inferred_model_type: {inferred_type}")
        
        # Try to match detections
        print(f"\n{'='*70}")
        print("ATTEMPTING TO MATCH DETECTIONS WITH GROUND TRUTH")
        print(f"{'='*70}")
        
        # Convert detections to pred_boxes format
        pred_boxes = []
        for det in detections:
            if det['confidence'] >= 0.4:  # Use default conf threshold
                pred_boxes.append({
                    'bbox': det['bbox'],
                    'class': det['class_name'],
                    'model_type': det['model_type'],
                    'confidence': det['confidence']
                })
        
        # Match
        matches, unmatched_preds, unmatched_gts = pipeline._match_detections(
            pred_boxes, gt_boxes, iou_threshold=0.5
        )
        
        print(f"\nMatching Results:")
        print(f"  Matches (TP): {len(matches)}")
        print(f"  Unmatched Predictions (FP): {len(unmatched_preds)}")
        print(f"  Unmatched Ground Truth (FN): {len(unmatched_gts)}")
        
        # Show match details
        if matches:
            print(f"\nMatched Detections:")
            for match in matches:
                pred = match['pred']
                gt = match['gt']
                iou = match['iou']
                print(f"  Prediction: {pred['model_type']} ({pred['confidence']:.2f})")
                print(f"  Ground Truth: {gt.get('class_name', 'N/A')} (inferred: {pipeline._infer_model_type_from_gt(gt)})")
                print(f"  IoU: {iou:.4f}")
                print()
        
        # Show unmatched predictions
        if unmatched_preds:
            print(f"\nUnmatched Predictions (False Positives):")
            for pred in unmatched_preds[:5]:  # Show first 5
                print(f"  {pred['model_type']} - confidence: {pred['confidence']:.2f}, bbox: {pred['bbox']}")
                # Check why it didn't match
                best_iou = 0.0
                best_gt_type = None
                for gt in gt_boxes:
                    iou = pipeline._calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_type = pipeline._infer_model_type_from_gt(gt)
                print(f"    Best IoU with any GT: {best_iou:.4f}")
                print(f"    Best GT model_type: {best_gt_type}")
                print(f"    Prediction model_type: {pred['model_type']}")
                if pred['model_type'] != best_gt_type:
                    print(f"    ⚠ MODEL TYPE MISMATCH!")
                print()
        
        # Show unmatched ground truth
        if unmatched_gts:
            print(f"\nUnmatched Ground Truth (False Negatives):")
            for gt in unmatched_gts[:5]:  # Show first 5
                inferred_type = pipeline._infer_model_type_from_gt(gt)
                print(f"  {gt.get('class_name', 'N/A')} (inferred: {inferred_type})")
                print(f"    bbox: {gt.get('bbox', 'N/A')}")
                print()
    else:
        print(f"\nNo ground truth label file found at: {label_path}")
        print("This explains why FN=0 (no ground truth to match against)")
        print("\nTo fix:")
        print("1. Ensure label files exist in the labels directory")
        print("2. Check that label files have the same name as images (e.g., image.jpg -> image.txt)")
        print("3. Verify label files are in YOLO format: class_id x_center y_center width height (normalized)")
    
    print(f"\n{'='*70}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug lane detection evaluation')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--label', type=str, default=None, help='Path to ground truth label file')
    parser.add_argument('--model', type=str, default='lane', help='Model to use')
    
    args = parser.parse_args()
    
    debug_single_image(args.image, args.label, args.model)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print("Usage: python debug_evaluation.py --image path/to/image.jpg [--label path/to/label.txt]")
        print("\nExample:")
        print("  python debug_evaluation.py --image lane_det/dataset/test/image.jpg --label lane_det/dataset/test/labels/image.txt")

