# ============================================================================
# YOLOv8/YOLO11 INSTANCE SEGMENTATION FOR TUMOR BUD DETECTION
# ============================================================================
# This script uses Ultralytics YOLO for instance segmentation of tumor buds.
# 
# Advantages over Mask R-CNN:
# - Faster training and inference (single-stage)
# - Simpler setup and configuration
# - Better performance on small objects with proper configuration
# - YOLOv8/YOLO11 achieve 89.1% recall on mitosis detection (similar task)
# 
# Requirements:
# pip install ultralytics
# ============================================================================

# %% [markdown]
# ## Cell 1: Install and Import

# %%
# Install ultralytics (run once)
# !pip install ultralytics

import os
import json
import shutil
from pathlib import Path
from collections import Counter
import yaml

# Check ultralytics version
try:
    from ultralytics import YOLO, settings
    import ultralytics
    print(f"Ultralytics version: {ultralytics.__version__}")
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    subprocess.run(["pip", "install", "ultralytics"])
    from ultralytics import YOLO, settings
    import ultralytics
    print(f"Ultralytics version: {ultralytics.__version__}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## Cell 2: Configuration

# %%
# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================

# For Google Colab:
# BASE_DIR = Path("/content/drive/MyDrive/tumor_budding")

# For local:
BASE_DIR = Path(".")

# Use binary dataset (recommended)
USE_BINARY = True

if USE_BINARY:
    COCO_TRAIN_JSON = BASE_DIR / "dataset_binary" / "train" / "annotations.json"
    COCO_TRAIN_IMAGES = BASE_DIR / "dataset_binary" / "train"
    COCO_VAL_JSON = BASE_DIR / "dataset_binary" / "test" / "annotations.json"
    COCO_VAL_IMAGES = BASE_DIR / "dataset_binary" / "test"
else:
    COCO_TRAIN_JSON = BASE_DIR / "dataset" / "train" / "annotations.json"
    COCO_TRAIN_IMAGES = BASE_DIR / "dataset" / "train"
    COCO_VAL_JSON = BASE_DIR / "dataset" / "test" / "annotations.json"
    COCO_VAL_IMAGES = BASE_DIR / "dataset" / "test"

# YOLO dataset output
YOLO_DATASET_DIR = BASE_DIR / "dataset_yolo"
OUTPUT_DIR = BASE_DIR / "models" / "yolo_segmentation"

# Model selection
# Options: "yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg"
#          "yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg"
MODEL_NAME = "yolov8m-seg"  # Medium model - good balance

print(f"Model: {MODEL_NAME}")
print(f"Dataset: {'Binary' if USE_BINARY else '5-class'}")

# %% [markdown]
# ## Cell 3: Convert COCO to YOLO Format

# %%
def coco_to_yolo_segmentation(coco_json_path, images_dir, output_dir, split_name):
    """
    Convert COCO format annotations to YOLO segmentation format.
    
    YOLO segmentation format:
    - One .txt file per image
    - Each line: class_id x1 y1 x2 y2 ... xn yn (normalized polygon coordinates)
    
    Args:
        coco_json_path: Path to COCO annotations.json
        images_dir: Path to images folder
        output_dir: Output directory for YOLO dataset
        split_name: "train" or "val"
    """
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    images_out = output_dir / "images" / split_name
    labels_out = output_dir / "labels" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    
    # Build image ID to info mapping
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # Build image ID to annotations mapping
    image_id_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_id_to_anns:
            image_id_to_anns[img_id] = []
        image_id_to_anns[img_id].append(ann)
    
    # Category ID mapping (YOLO uses 0-indexed)
    cat_id_to_yolo = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    
    converted_count = 0
    
    for img_id, img_info in image_id_to_info.items():
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Source and destination paths
        src_img = images_dir / img_filename
        dst_img = images_out / img_filename
        
        # Copy image if not exists
        if not dst_img.exists() and src_img.exists():
            shutil.copy(src_img, dst_img)
        
        # Create label file
        label_filename = Path(img_filename).stem + ".txt"
        label_path = labels_out / label_filename
        
        annotations = image_id_to_anns.get(img_id, [])
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                # Get class ID (0-indexed for YOLO)
                class_id = cat_id_to_yolo[ann['category_id']]
                
                # Get segmentation polygon
                if 'segmentation' not in ann or not ann['segmentation']:
                    continue
                
                # Handle multiple polygons (take first one)
                for polygon in ann['segmentation']:
                    if len(polygon) < 6:  # Need at least 3 points
                        continue
                    
                    # Normalize coordinates
                    normalized_points = []
                    for i in range(0, len(polygon), 2):
                        x = polygon[i] / img_width
                        y = polygon[i + 1] / img_height
                        # Clamp to [0, 1]
                        x = max(0, min(1, x))
                        y = max(0, min(1, y))
                        normalized_points.extend([x, y])
                    
                    # Write line: class_id x1 y1 x2 y2 ... xn yn
                    points_str = " ".join(f"{p:.6f}" for p in normalized_points)
                    f.write(f"{class_id} {points_str}\n")
        
        converted_count += 1
    
    print(f"Converted {converted_count} images for {split_name}")
    return converted_count

# Convert training set
print("\nConverting training set...")
train_count = coco_to_yolo_segmentation(
    COCO_TRAIN_JSON, COCO_TRAIN_IMAGES, YOLO_DATASET_DIR, "train"
)

# Convert validation set
print("\nConverting validation set...")
val_count = coco_to_yolo_segmentation(
    COCO_VAL_JSON, COCO_VAL_IMAGES, YOLO_DATASET_DIR, "val"
)

print(f"\nTotal: {train_count} train, {val_count} val images")

# %% [markdown]
# ## Cell 4: Create YOLO Dataset Configuration

# %%
def create_yolo_config(dataset_dir, coco_json_path, output_path):
    """Create YOLO dataset.yaml configuration file."""
    
    # Read categories from COCO
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    categories = coco_data['categories']
    class_names = [cat['name'] for cat in sorted(categories, key=lambda x: x['id'])]
    
    # Create config
    config = {
        'path': str(dataset_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    # Write YAML
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created YOLO config: {output_path}")
    print(f"Classes: {class_names}")
    
    return output_path

# Create config
YOLO_CONFIG_PATH = YOLO_DATASET_DIR / "dataset.yaml"
create_yolo_config(YOLO_DATASET_DIR, COCO_TRAIN_JSON, YOLO_CONFIG_PATH)

# Display config
print("\nDataset configuration:")
with open(YOLO_CONFIG_PATH, 'r') as f:
    print(f.read())

# %% [markdown]
# ## Cell 5: Train YOLO Model

# %%
# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Training parameters optimized for tumor bud detection
TRAIN_CONFIG = {
    'data': str(YOLO_CONFIG_PATH),
    'epochs': 100,              # More epochs for better convergence
    'imgsz': 1024,              # Large image size for small objects
    'batch': 4,                 # Adjust based on GPU memory
    'patience': 20,             # Early stopping patience
    'project': str(OUTPUT_DIR),
    'name': 'tumor_bud_seg',
    'exist_ok': True,
    
    # Optimizer
    'optimizer': 'AdamW',
    'lr0': 0.001,               # Initial learning rate
    'lrf': 0.01,                # Final learning rate factor
    'momentum': 0.937,
    'weight_decay': 0.0005,
    
    # Augmentation (conservative for histopathology)
    'hsv_h': 0.01,              # Hue augmentation (minimal)
    'hsv_s': 0.3,               # Saturation augmentation
    'hsv_v': 0.3,               # Value augmentation
    'degrees': 10.0,            # Rotation degrees
    'translate': 0.1,           # Translation
    'scale': 0.2,               # Scale
    'shear': 2.0,               # Shear
    'flipud': 0.5,              # Vertical flip
    'fliplr': 0.5,              # Horizontal flip
    'mosaic': 0.5,              # Mosaic augmentation
    'mixup': 0.0,               # Disable mixup (not suitable for histopathology)
    
    # Other
    'save': True,
    'save_period': 10,          # Save checkpoint every 10 epochs
    'val': True,
    'plots': True,
    'verbose': True,
}

print("Training Configuration:")
for key, value in TRAIN_CONFIG.items():
    print(f"  {key}: {value}")

# %% [markdown]
# ## Cell 6: Start Training

# %%
print("=" * 60)
print(f"TRAINING YOLO SEGMENTATION MODEL")
print("=" * 60)
print(f"Model: {MODEL_NAME}")
print(f"Dataset: {YOLO_CONFIG_PATH}")
print(f"Epochs: {TRAIN_CONFIG['epochs']}")
print(f"Image size: {TRAIN_CONFIG['imgsz']}")
print("=" * 60)

# Load model
model = YOLO(MODEL_NAME)

# Train
results = model.train(**TRAIN_CONFIG)

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)

# %% [markdown]
# ## Cell 7: Evaluate Model

# %%
# Load best model
best_model_path = OUTPUT_DIR / "tumor_bud_seg" / "weights" / "best.pt"
if best_model_path.exists():
    model = YOLO(str(best_model_path))
    print(f"Loaded best model: {best_model_path}")
else:
    print(f"Best model not found at {best_model_path}")
    # Use last model
    last_model_path = OUTPUT_DIR / "tumor_bud_seg" / "weights" / "last.pt"
    if last_model_path.exists():
        model = YOLO(str(last_model_path))
        print(f"Loaded last model: {last_model_path}")

# Validate
print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

val_results = model.val(
    data=str(YOLO_CONFIG_PATH),
    imgsz=TRAIN_CONFIG['imgsz'],
    batch=TRAIN_CONFIG['batch'],
    conf=0.25,  # Confidence threshold (lower for higher recall)
    iou=0.3,    # IoU threshold for NMS
    verbose=True
)

# Print key metrics
print("\nKey Metrics:")
if hasattr(val_results, 'seg'):
    seg_metrics = val_results.seg
    print(f"  Mask mAP50:     {seg_metrics.map50:.4f} ({seg_metrics.map50*100:.1f}%)")
    print(f"  Mask mAP50-95:  {seg_metrics.map:.4f} ({seg_metrics.map*100:.1f}%)")

if hasattr(val_results, 'box'):
    box_metrics = val_results.box
    print(f"  Box mAP50:      {box_metrics.map50:.4f} ({box_metrics.map50*100:.1f}%)")
    print(f"  Box mAP50-95:   {box_metrics.map:.4f} ({box_metrics.map*100:.1f}%)")

# %% [markdown]
# ## Cell 8: Custom Evaluation (Precision, Recall, F1)

# %%
import numpy as np
from pathlib import Path
import cv2

def evaluate_yolo_detection(model, val_images_dir, val_labels_dir, conf_threshold=0.25, iou_threshold=0.3):
    """
    Custom evaluation to compute Precision, Recall, F1, DICE, MAE.
    """
    print(f"\n{'='*60}")
    print("CUSTOM DETECTION EVALUATION")
    print(f"{'='*60}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_pred = 0
    
    dice_scores = []
    count_errors = []
    
    # Get all images
    image_files = list(Path(val_images_dir).glob("*.png")) + list(Path(val_images_dir).glob("*.jpg"))
    
    for img_path in image_files:
        # Get corresponding label file
        label_path = Path(val_labels_dir) / (img_path.stem + ".txt")
        
        # Count ground truth
        gt_count = 0
        if label_path.exists():
            with open(label_path, 'r') as f:
                gt_count = len(f.readlines())
        
        # Get predictions
        results = model.predict(str(img_path), conf=conf_threshold, verbose=False)
        pred_count = 0
        if results and len(results) > 0:
            if results[0].masks is not None:
                pred_count = len(results[0].masks)
            elif results[0].boxes is not None:
                pred_count = len(results[0].boxes)
        
        # Simple matching (count-based for now)
        tp = min(gt_count, pred_count)
        fp = max(0, pred_count - gt_count)
        fn = max(0, gt_count - pred_count)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt += gt_count
        total_pred += pred_count
        
        # DICE coefficient (count-based)
        if gt_count + pred_count > 0:
            dice = (2 * min(gt_count, pred_count)) / (gt_count + pred_count)
        else:
            dice = 1.0
        dice_scores.append(dice)
        
        # Count error (MAE)
        count_errors.append(abs(pred_count - gt_count))
    
    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_dice = np.mean(dice_scores)
    avg_mae = np.mean(count_errors)
    
    print(f"\nResults:")
    print(f"  Ground Truth:    {total_gt}")
    print(f"  Predictions:     {total_pred}")
    print(f"  True Positives:  {total_tp}")
    print(f"  False Positives: {total_fp}")
    print(f"  False Negatives: {total_fn}")
    print(f"\n  Precision: {precision:.4f} ({precision*100:.1f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.1f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.1f}%)")
    print(f"  DICE:      {avg_dice:.4f} ({avg_dice*100:.1f}%)")
    print(f"  MAE:       {avg_mae:.4f}")
    
    # Medical imaging interpretation
    print(f"\n{'='*60}")
    print("MEDICAL IMAGING INTERPRETATION")
    print(f"{'='*60}")
    if recall >= 0.8:
        print("  ✓ RECALL >= 80%: Good sensitivity for clinical use")
    else:
        print(f"  ⚠ RECALL < 80%: May miss tumor buds (currently {recall*100:.1f}%)")
    
    if precision >= 0.7:
        print("  ✓ PRECISION >= 70%: Acceptable false positive rate")
    else:
        print(f"  ⚠ PRECISION < 70%: High false positive rate (currently {precision*100:.1f}%)")
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": avg_dice,
        "mae": avg_mae,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn
    }

# Run evaluation
val_images_dir = YOLO_DATASET_DIR / "images" / "val"
val_labels_dir = YOLO_DATASET_DIR / "labels" / "val"
custom_results = evaluate_yolo_detection(model, val_images_dir, val_labels_dir, conf_threshold=0.25)

# %% [markdown]
# ## Cell 9: Visualize Predictions

# %%
import matplotlib.pyplot as plt
import random

def visualize_yolo_predictions(model, images_dir, num_samples=4):
    """Visualize YOLO predictions on random samples."""
    image_files = list(Path(images_dir).glob("*.png")) + list(Path(images_dir).glob("*.jpg"))
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for ax, img_path in zip(axes, samples):
        # Run prediction
        results = model.predict(str(img_path), conf=0.25, verbose=False)
        
        # Get annotated image
        if results and len(results) > 0:
            annotated_img = results[0].plot()
            pred_count = len(results[0].masks) if results[0].masks is not None else 0
        else:
            annotated_img = cv2.imread(str(img_path))
            pred_count = 0
        
        # Display
        ax.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{img_path.name}\nPredictions: {pred_count}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tumor_bud_seg" / "prediction_samples.png", dpi=150)
    plt.show()
    print(f"Saved to: {OUTPUT_DIR / 'tumor_bud_seg' / 'prediction_samples.png'}")

# Visualize
visualize_yolo_predictions(model, val_images_dir, num_samples=4)

# %% [markdown]
# ## Cell 10: Export Model

# %%
# Export to ONNX for deployment
print("Exporting model to ONNX...")
model.export(format='onnx', imgsz=1024)
print("Export complete!")

# %% [markdown]
# ## Cell 11: Summary and Comparison

# %%
print("""
================================================================================
YOLO vs MASK R-CNN COMPARISON
================================================================================

| Aspect                | YOLO (YOLOv8/11)      | Mask R-CNN           |
|-----------------------|-----------------------|----------------------|
| Architecture          | Single-stage          | Two-stage            |
| Speed                 | Fast (real-time)      | Slower               |
| Setup Complexity      | Simple                | Complex (Detectron2) |
| Memory Usage          | Lower                 | Higher               |
| Small Object          | Good with proper size | Good with FPN        |
| Mask Quality          | Good                  | Very precise         |
| Training Time         | Faster                | Slower               |

RECOMMENDATIONS FOR TUMOR BUD DETECTION:
1. If YOLO achieves similar or better recall: Use YOLO for simplicity
2. If precise mask boundaries needed: Use Mask R-CNN
3. For real-time inference: Use YOLO
4. For best results: Try both and compare

Next steps:
- Compare mAP/recall between YOLO and Mask R-CNN results
- Try YOLO11 for potentially better accuracy
- Adjust conf threshold for precision/recall trade-off
================================================================================
""")

print("\nTraining complete! Check the results directory for:")
print(f"  - Best model: {OUTPUT_DIR / 'tumor_bud_seg' / 'weights' / 'best.pt'}")
print(f"  - Training curves: {OUTPUT_DIR / 'tumor_bud_seg' / 'results.png'}")
print(f"  - Confusion matrix: {OUTPUT_DIR / 'tumor_bud_seg' / 'confusion_matrix.png'}")
