# ============================================================================
# BINARY CLASSIFICATION TRAINING - TUMOR BUD DETECTION
# ============================================================================
# This script trains a Mask R-CNN model for binary tumor bud detection.
# Uses the converted binary dataset (all 5 classes merged into "tumor_bud").
# 
# Key optimizations:
# - Single class = no class imbalance issues
# - 2000+ training samples
# - Conservative augmentations for histopathology
# - Optimized for recall (minimize False Negatives)
# ============================================================================

# %% [markdown]
# ## Cell 1: Setup and Install Dependencies (Google Colab)

# %%
# Uncomment these lines if running in Google Colab
# from google.colab import drive
# drive.mount('/content/drive', force_remount=False)

# # Install detectron2 (Colab method)
# !python -m pip install pyyaml==5.1
# import sys, os, distutils.core
# !git clone 'https://github.com/facebookresearch/detectron2'
# dist = distutils.core.run_setup("./detectron2/setup.py")
# !python -m pip install {' '.join([f"'{x}'" for x in dist.install_requires])}
# sys.path.insert(0, os.path.abspath('./detectron2'))

# %% [markdown]
# ## Cell 2: Imports

# %%
import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os
import json
import cv2
import random
import yaml
import pandas as pd
from collections import Counter
from pathlib import Path
from matplotlib import pyplot as plt
import copy

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode

import albumentations as A

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# %% [markdown]
# ## Cell 3: Configuration - UPDATE THESE PATHS

# %%
# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR ENVIRONMENT
# =============================================================================

# For Google Colab:
# BASE_DIR = Path("/content/drive/MyDrive/tumor_budding")

# For local:
BASE_DIR = Path(".")

# Binary dataset paths (created by convert_to_binary.py)
TRAIN_JSON = str(BASE_DIR / "dataset_binary" / "train" / "annotations.json")
TRAIN_IMAGES = str(BASE_DIR / "dataset_binary" / "train")
VAL_JSON = str(BASE_DIR / "dataset_binary" / "test" / "annotations.json")
VAL_IMAGES = str(BASE_DIR / "dataset_binary" / "test")

# Output directory for models
OUTPUT_DIR = str(BASE_DIR / "models" / "binary_maskrcnn")

# Verify paths exist
for path in [TRAIN_JSON, TRAIN_IMAGES, VAL_JSON, VAL_IMAGES]:
    if not Path(path).exists():
        print(f"WARNING: Path does not exist: {path}")
        print("Make sure to run convert_to_binary.py first!")
    else:
        print(f"✓ Found: {path}")

# %% [markdown]
# ## Cell 4: Register Dataset

# %%
# Register datasets (only if not already registered)
if "tumor_bud_binary_train" in DatasetCatalog.list():
    DatasetCatalog.remove("tumor_bud_binary_train")
    MetadataCatalog.remove("tumor_bud_binary_train")
    
if "tumor_bud_binary_val" in DatasetCatalog.list():
    DatasetCatalog.remove("tumor_bud_binary_val")
    MetadataCatalog.remove("tumor_bud_binary_val")

register_coco_instances("tumor_bud_binary_train", {}, TRAIN_JSON, TRAIN_IMAGES)
register_coco_instances("tumor_bud_binary_val", {}, VAL_JSON, VAL_IMAGES)

# Get metadata and dataset
train_metadata = MetadataCatalog.get("tumor_bud_binary_train")
train_dataset = DatasetCatalog.get("tumor_bud_binary_train")
val_metadata = MetadataCatalog.get("tumor_bud_binary_val")
val_dataset = DatasetCatalog.get("tumor_bud_binary_val")

print(f"\nDataset Statistics:")
print(f"  Classes: {train_metadata.thing_classes}")
print(f"  Training images: {len(train_dataset)}")
print(f"  Validation images: {len(val_dataset)}")

# Count annotations
train_ann_count = sum(len(d['annotations']) for d in train_dataset)
val_ann_count = sum(len(d['annotations']) for d in val_dataset)
print(f"  Training annotations: {train_ann_count}")
print(f"  Validation annotations: {val_ann_count}")

# %% [markdown]
# ## Cell 5: Optimized Augmentation Pipeline

# %%
def get_conservative_augmentation():
    """
    Conservative augmentation pipeline for histopathology images.
    
    REMOVED (can hurt accuracy):
    - GaussianBlur, MotionBlur (WSI images are always sharp)
    - Heavy ElasticTransform (distorts cell morphology)
    - Heavy Affine (changes cell sizes too much)
    
    KEPT (beneficial):
    - Flips and 90° rotations (cells have no orientation)
    - Mild color adjustments (handles staining variation)
    - CLAHE (enhances contrast for faint cells)
    """
    transform = A.Compose([
        # === GEOMETRIC (Safe for cells) ===
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # Slight rotation (pathology has no orientation)
        A.Rotate(limit=10, p=0.3, border_mode=cv2.BORDER_REFLECT),
        
        # === COLOR/INTENSITY (Handles staining variation) ===
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=8,
                sat_shift_limit=15,
                val_shift_limit=15,
                p=1.0
            ),
        ], p=0.4),
        
        # Mild color jitter for scanner variation
        A.ColorJitter(
            brightness=0.08,
            contrast=0.08,
            saturation=0.08,
            hue=0.03,
            p=0.3
        ),
        
        # CLAHE for enhancing faint cells
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.15),
        
        # === NO BLUR (WSI images are sharp) ===
        # GaussianBlur, MotionBlur, MedianBlur - REMOVED
        
        # === MINIMAL NOISE (only very slight) ===
        A.GaussNoise(std_range=(0.01, 0.02), p=0.1),  # Very mild
        
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['category_ids'],
        min_area=1,
        min_visibility=0.3,  # Keep boxes that are at least 30% visible
    ))
    
    return transform


class ConservativeAugMapper:
    """
    Custom DatasetMapper with conservative augmentations for histopathology.
    """
    
    def __init__(self, cfg, is_train=True):
        self.cfg = cfg
        self.is_train = is_train
        self.augmentation = get_conservative_augmentation() if is_train else None
        self.image_format = cfg.INPUT.FORMAT
        
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        
        # Load image
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        annotations = dataset_dict.get("annotations", [])
        
        if self.is_train and self.augmentation and len(annotations) > 0:
            # Prepare data for albumentations
            bboxes = []
            category_ids = []
            masks = []
            
            for ann in annotations:
                bbox = ann["bbox"]
                if ann.get("bbox_mode") == BoxMode.XYXY_ABS:
                    bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                bboxes.append(bbox)
                category_ids.append(ann["category_id"])
                
                # Convert polygon to binary mask
                h, w = image.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                for poly in ann.get("segmentation", []):
                    pts = np.array(poly).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
                masks.append(mask)
            
            try:
                # Convert BGR to RGB for albumentations
                if self.image_format == "BGR":
                    image_rgb = image[:, :, ::-1]
                else:
                    image_rgb = image
                
                transformed = self.augmentation(
                    image=image_rgb,
                    bboxes=bboxes,
                    category_ids=category_ids,
                    masks=masks
                )
                
                image_aug = transformed["image"]
                bboxes_aug = transformed["bboxes"]
                category_ids_aug = transformed["category_ids"]
                masks_aug = transformed["masks"]
                
                # Convert back to BGR
                if self.image_format == "BGR":
                    image_aug = image_aug[:, :, ::-1]
                
                image = image_aug
                
                # Rebuild annotations
                new_annotations = []
                for bbox, cat_id, mask_aug in zip(bboxes_aug, category_ids_aug, masks_aug):
                    contours, _ = cv2.findContours(
                        mask_aug.astype(np.uint8), 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    segmentation = []
                    for contour in contours:
                        if contour.size >= 6:
                            segmentation.append(contour.flatten().tolist())
                    
                    if segmentation:
                        new_annotations.append({
                            "bbox": list(bbox),
                            "bbox_mode": BoxMode.XYWH_ABS,
                            "category_id": cat_id,
                            "segmentation": segmentation,
                            "iscrowd": 0,
                        })
                
                annotations = new_annotations
                
            except Exception as e:
                print(f"Augmentation failed: {e}")
                pass
        
        # Convert to tensor
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        
        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict
        
        # Create instances
        annos = [
            utils.transform_instance_annotations(obj, [], image.shape[:2])
            for obj in annotations
        ]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        
        return dataset_dict


class ConservativeTrainer(DefaultTrainer):
    """Trainer with conservative augmentations."""
    
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = ConservativeAugMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

# %% [markdown]
# ## Cell 6: Model Configuration (Optimized for Binary Detection)

# %%
cfg = get_cfg()

# Use R50-FPN as base (good balance of speed and accuracy)
# For better accuracy, try: "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)

# === DATASET ===
cfg.DATASETS.TRAIN = ("tumor_bud_binary_train",)
cfg.DATASETS.TEST = ("tumor_bud_binary_val",)
cfg.DATALOADER.NUM_WORKERS = 2

# === OUTPUT ===
cfg.OUTPUT_DIR = OUTPUT_DIR
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# === SOLVER (Training Parameters) ===
cfg.SOLVER.IMS_PER_BATCH = 4  # Batch size (reduce to 2 if OOM)
cfg.SOLVER.BASE_LR = 0.0025   # Learning rate
cfg.SOLVER.MAX_ITER = 8000    # More iterations for better convergence
cfg.SOLVER.STEPS = (5000, 7000)  # LR decay steps
cfg.SOLVER.GAMMA = 0.5       # LR decay factor
cfg.SOLVER.WARMUP_ITERS = 500  # Warmup iterations
cfg.SOLVER.WARMUP_FACTOR = 1.0 / 500

# === MODEL (Binary Classification) ===
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Single class: tumor_bud
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

# === SMALL OBJECT OPTIMIZATION ===
# Smaller anchors for small tumor buds
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

# RPN settings for small objects
cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
cfg.MODEL.RPN.NMS_THRESH = 0.7
cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]

# ROI Heads (lower threshold to catch more small objects)
cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.4]

# === EVALUATION (Optimized for Recall) ===
# Lower score threshold = higher recall (catch more tumor buds)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Low for high recall
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4    # Allow some overlap

print("Configuration Summary:")
print(f"  Model: R50-FPN")
print(f"  Classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
print(f"  Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
print(f"  Learning rate: {cfg.SOLVER.BASE_LR}")
print(f"  Max iterations: {cfg.SOLVER.MAX_ITER}")
print(f"  Anchor sizes: {cfg.MODEL.ANCHOR_GENERATOR.SIZES}")
print(f"  Score threshold: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")
print(f"  Output: {cfg.OUTPUT_DIR}")

# %% [markdown]
# ## Cell 7: Visualize Sample Data

# %%
def visualize_samples(dataset, metadata, num_samples=3):
    """Visualize random samples from the dataset."""
    fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
    if num_samples == 1:
        axes = [axes]
    
    for ax, d in zip(axes, random.sample(dataset, min(num_samples, len(dataset)))):
        img = cv2.imread(d["file_name"])
        if img is None:
            continue
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        # Note: Visualizer.get_image() returns RGB, don't convert again!
        ax.imshow(vis.get_image())
        ax.set_title(f"{os.path.basename(d['file_name'])}\n{len(d['annotations'])} annotations")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize training samples
print("Sample training images with binary annotations:")
visualize_samples(train_dataset, train_metadata, num_samples=3)

# %% [markdown]
# ## Cell 8: Train the Model

# %%
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)
print(f"Model: Mask R-CNN R50-FPN")
print(f"Dataset: Binary tumor bud detection")
print(f"Training images: {len(train_dataset)}")
print(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
print("=" * 60)

trainer = ConservativeTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"Model saved to: {cfg.OUTPUT_DIR}")

# %% [markdown]
# ## Cell 9: Save Configuration

# %%
# Save the full configuration for reproducibility
config_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
with open(config_path, 'w') as f:
    f.write(cfg.dump())
print(f"Configuration saved to: {config_path}")

# Also save a summary
summary_path = os.path.join(cfg.OUTPUT_DIR, "training_summary.txt")
with open(summary_path, 'w') as f:
    f.write("Binary Tumor Bud Detection - Training Summary\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Model: Mask R-CNN R50-FPN\n")
    f.write(f"Classes: 1 (tumor_bud)\n")
    f.write(f"Training images: {len(train_dataset)}\n")
    f.write(f"Training annotations: {train_ann_count}\n")
    f.write(f"Validation images: {len(val_dataset)}\n")
    f.write(f"Validation annotations: {val_ann_count}\n\n")
    f.write(f"Hyperparameters:\n")
    f.write(f"  Batch size: {cfg.SOLVER.IMS_PER_BATCH}\n")
    f.write(f"  Learning rate: {cfg.SOLVER.BASE_LR}\n")
    f.write(f"  Max iterations: {cfg.SOLVER.MAX_ITER}\n")
    f.write(f"  Anchor sizes: {cfg.MODEL.ANCHOR_GENERATOR.SIZES}\n")
    f.write(f"  Score threshold: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}\n")
print(f"Summary saved to: {summary_path}")

# %% [markdown]
# ## Cell 10: Load Trained Model for Evaluation

# %%
# Load the trained model
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
predictor = DefaultPredictor(cfg)
print(f"Loaded model from: {cfg.MODEL.WEIGHTS}")

# %% [markdown]
# ## Cell 11: Evaluation Functions

# %%
def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def evaluate_binary_detection(dataset, predictor, iou_threshold=0.3):
    """
    Evaluate binary tumor bud detection.
    
    Metrics computed:
    - Precision, Recall, F1-Score (detection performance)
    - DICE coefficient (count-based)
    - MAE (counting error)
    - Detection Rate (sensitivity)
    - Specificity (for completeness)
    
    Args:
        dataset: Validation dataset
        predictor: Trained predictor
        iou_threshold: IoU threshold for matching (0.3 for medical imaging)
    
    Returns:
        dict with all metrics
    """
    print(f"\n{'='*60}")
    print("BINARY TUMOR BUD DETECTION EVALUATION")
    print(f"{'='*60}")
    print(f"IoU threshold: {iou_threshold}")
    print(f"Evaluating {len(dataset)} images...\n")
    
    total_tp = 0  # True Positives
    total_fp = 0  # False Positives
    total_fn = 0  # False Negatives
    total_gt = 0  # Total ground truth
    total_pred = 0  # Total predictions
    
    dice_scores = []
    count_errors = []
    
    for d in dataset:
        # Ground truth
        gt_annotations = d["annotations"]
        gt_boxes = []
        for ann in gt_annotations:
            bbox = ann["bbox"]
            # Convert [x, y, w, h] to [x1, y1, x2, y2]
            if len(bbox) == 4:
                gt_boxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
        
        # Predictions
        img = cv2.imread(d["file_name"])
        if img is None:
            continue
        
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        pred_boxes = instances.pred_boxes.tensor.numpy().tolist() if len(instances) > 0 else []
        
        # Match predictions to ground truth
        matched_gt = set()
        matched_pred = set()
        
        for pred_idx, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx != -1:
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
        
        tp = len(matched_pred)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - len(matched_gt)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        
        # DICE for this image
        gt_count = len(gt_boxes)
        pred_count = len(pred_boxes)
        if gt_count + pred_count > 0:
            dice = (2 * min(gt_count, pred_count)) / (gt_count + pred_count)
        else:
            dice = 1.0
        dice_scores.append(dice)
        
        # Count error
        count_errors.append(abs(pred_count - gt_count))
    
    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    avg_dice = np.mean(dice_scores)
    avg_mae = np.mean(count_errors)
    detection_rate = recall  # Same as recall
    
    # Print results
    print(f"{'='*60}")
    print("DETECTION METRICS")
    print(f"{'='*60}")
    print(f"  True Positives:  {total_tp}")
    print(f"  False Positives: {total_fp}")
    print(f"  False Negatives: {total_fn}")
    print(f"  Ground Truth:    {total_gt}")
    print(f"  Predictions:     {total_pred}")
    print()
    print(f"{'='*60}")
    print("PERFORMANCE METRICS")
    print(f"{'='*60}")
    print(f"  Precision:       {precision:.4f} ({precision*100:.1f}%)")
    print(f"  Recall:          {recall:.4f} ({recall*100:.1f}%)")
    print(f"  F1-Score:        {f1:.4f} ({f1*100:.1f}%)")
    print(f"  DICE:            {avg_dice:.4f} ({avg_dice*100:.1f}%)")
    print(f"  MAE:             {avg_mae:.4f}")
    print(f"  Detection Rate:  {detection_rate:.4f} ({detection_rate*100:.1f}%)")
    print(f"{'='*60}")
    
    # Medical imaging interpretation
    print("\nMEDICAL IMAGING INTERPRETATION:")
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
        "detection_rate": detection_rate,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "total_gt": total_gt,
        "total_pred": total_pred
    }

# %% [markdown]
# ## Cell 12: Run Evaluation

# %%
# Run evaluation on validation set
results = evaluate_binary_detection(val_dataset, predictor, iou_threshold=0.3)

# Save results
results_path = os.path.join(cfg.OUTPUT_DIR, "evaluation_results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {results_path}")

# %% [markdown]
# ## Cell 13: Visualize Predictions

# %%
def visualize_predictions(dataset, predictor, metadata, num_samples=4):
    """Visualize predictions on random samples."""
    samples = random.sample(dataset, min(num_samples, len(dataset)))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for ax, d in zip(axes, samples):
        img = cv2.imread(d["file_name"])
        if img is None:
            continue
        
        outputs = predictor(img)
        pred_count = len(outputs["instances"])
        gt_count = len(d["annotations"])
        
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.6)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # Note: Visualizer.get_image() returns RGB, don't convert again!
        ax.imshow(out.get_image())
        ax.set_title(f"GT: {gt_count} | Pred: {pred_count}", fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "prediction_samples.png"), dpi=150)
    plt.show()
    print(f"Saved to: {os.path.join(cfg.OUTPUT_DIR, 'prediction_samples.png')}")

# Visualize predictions
visualize_predictions(val_dataset, predictor, val_metadata, num_samples=4)

# %% [markdown]
# ## Cell 14: Summary

# %%
print("\n" + "=" * 60)
print("BINARY CLASSIFICATION TRAINING COMPLETE")
print("=" * 60)
print(f"\nModel: Mask R-CNN R50-FPN (Binary)")
print(f"Classes: 1 (tumor_bud)")
print(f"\nFinal Metrics:")
print(f"  Precision: {results['precision']*100:.1f}%")
print(f"  Recall:    {results['recall']*100:.1f}%")
print(f"  F1-Score:  {results['f1']*100:.1f}%")
print(f"  DICE:      {results['dice']*100:.1f}%")
print(f"\nFiles saved:")
print(f"  Model: {os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')}")
print(f"  Config: {config_path}")
print(f"  Results: {results_path}")
print("=" * 60)
