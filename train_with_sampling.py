# ============================================================================
# 5-CLASS TRAINING WITH CLASS IMBALANCE HANDLING
# ============================================================================
# This script trains a Mask R-CNN model with 5 classes (1-cell to 5-cell)
# while handling severe class imbalance using:
# 1. RepeatFactorTrainingSampler - oversample images with rare classes
# 2. Conservative augmentations - avoid harmful transforms
# 3. Optimized hyperparameters for small objects
# 
# Use this if you need to differentiate between 1-cell, 2-cell, etc.
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
import copy
import math
from pathlib import Path
from collections import Counter, defaultdict
from matplotlib import pyplot as plt

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import (
    MetadataCatalog, 
    DatasetCatalog, 
    build_detection_train_loader,
    DatasetMapper
)
from detectron2.data.datasets import register_coco_instances
from detectron2.data import detection_utils as utils
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.structures import BoxMode

import albumentations as A

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# %% [markdown]
# ## Cell 3: Configuration

# %%
# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR ENVIRONMENT
# =============================================================================

# For Google Colab:
# BASE_DIR = Path("/content/drive/MyDrive/tumor_budding")

# For local:
BASE_DIR = Path(".")

# Original 5-class dataset
TRAIN_JSON = str(BASE_DIR / "dataset" / "train" / "annotations.json")
TRAIN_IMAGES = str(BASE_DIR / "dataset" / "train")
VAL_JSON = str(BASE_DIR / "dataset" / "test" / "annotations.json")
VAL_IMAGES = str(BASE_DIR / "dataset" / "test")

# Output directory
OUTPUT_DIR = str(BASE_DIR / "models" / "multiclass_balanced")

# Verify paths
for path in [TRAIN_JSON, TRAIN_IMAGES, VAL_JSON, VAL_IMAGES]:
    if Path(path).exists():
        print(f"✓ Found: {path}")
    else:
        print(f"✗ Missing: {path}")

# %% [markdown]
# ## Cell 4: Analyze Class Distribution

# %%
def analyze_class_distribution(json_path):
    """Analyze class distribution in annotations."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get category names
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Count annotations per class
    class_counts = Counter(ann['category_id'] for ann in data['annotations'])
    
    # Count images per class
    image_class_counts = defaultdict(set)
    for ann in data['annotations']:
        image_class_counts[ann['category_id']].add(ann['image_id'])
    
    print(f"\nClass Distribution in {Path(json_path).parent.name}:")
    print("-" * 50)
    print(f"{'Class':<15} {'Annotations':<15} {'Images':<15}")
    print("-" * 50)
    
    for cat_id in sorted(class_counts.keys()):
        name = cat_id_to_name.get(cat_id, f"Class {cat_id}")
        ann_count = class_counts[cat_id]
        img_count = len(image_class_counts[cat_id])
        print(f"{name:<15} {ann_count:<15} {img_count:<15}")
    
    print("-" * 50)
    print(f"{'Total':<15} {sum(class_counts.values()):<15} {len(data['images']):<15}")
    
    return class_counts, cat_id_to_name

train_counts, cat_names = analyze_class_distribution(TRAIN_JSON)
test_counts, _ = analyze_class_distribution(VAL_JSON)

# %% [markdown]
# ## Cell 5: Calculate Repeat Factors for Class Balancing

# %%
def calculate_repeat_factors(json_path, repeat_thresh=0.5):
    """
    Calculate repeat factors for RepeatFactorTrainingSampler.
    
    Images with rare classes will be repeated more often during training.
    
    Args:
        json_path: Path to COCO annotations
        repeat_thresh: Threshold for repeat factor calculation
                      Lower values = more aggressive oversampling
    
    Returns:
        dict mapping image_id to repeat_factor
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Count annotations per category
    category_counts = Counter(ann['category_id'] for ann in data['annotations'])
    total_annotations = sum(category_counts.values())
    
    # Calculate category frequency
    category_freq = {cat_id: count / total_annotations 
                    for cat_id, count in category_counts.items()}
    
    # Calculate repeat factors per category
    category_repeat = {}
    for cat_id, freq in category_freq.items():
        # Repeat factor formula from LVIS paper
        repeat_factor = max(1.0, math.sqrt(repeat_thresh / freq))
        category_repeat[cat_id] = repeat_factor
    
    print("\nRepeat Factors per Category:")
    print("-" * 40)
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    for cat_id in sorted(category_repeat.keys()):
        name = cat_id_to_name.get(cat_id, f"Class {cat_id}")
        freq = category_freq[cat_id]
        repeat = category_repeat[cat_id]
        print(f"  {name:<15} freq={freq:.4f}  repeat={repeat:.2f}x")
    
    # Calculate repeat factor per image (max of its categories)
    image_categories = defaultdict(set)
    for ann in data['annotations']:
        image_categories[ann['image_id']].add(ann['category_id'])
    
    image_repeat_factors = {}
    for img_id, categories in image_categories.items():
        max_repeat = max(category_repeat[cat_id] for cat_id in categories)
        image_repeat_factors[img_id] = max_repeat
    
    print(f"\nImages with repeat factor > 1:")
    repeat_counts = Counter(round(r, 1) for r in image_repeat_factors.values())
    for repeat, count in sorted(repeat_counts.items()):
        print(f"  {repeat:.1f}x: {count} images")
    
    return image_repeat_factors, category_repeat

image_repeat_factors, category_repeat = calculate_repeat_factors(TRAIN_JSON, repeat_thresh=0.3)

# %% [markdown]
# ## Cell 6: Register Dataset

# %%
# Register datasets
if "tumor_bud_5class_train" in DatasetCatalog.list():
    DatasetCatalog.remove("tumor_bud_5class_train")
    MetadataCatalog.remove("tumor_bud_5class_train")
    
if "tumor_bud_5class_val" in DatasetCatalog.list():
    DatasetCatalog.remove("tumor_bud_5class_val")
    MetadataCatalog.remove("tumor_bud_5class_val")

register_coco_instances("tumor_bud_5class_train", {}, TRAIN_JSON, TRAIN_IMAGES)
register_coco_instances("tumor_bud_5class_val", {}, VAL_JSON, VAL_IMAGES)

train_metadata = MetadataCatalog.get("tumor_bud_5class_train")
train_dataset = DatasetCatalog.get("tumor_bud_5class_train")
val_metadata = MetadataCatalog.get("tumor_bud_5class_val")
val_dataset = DatasetCatalog.get("tumor_bud_5class_val")

print(f"\nDataset registered:")
print(f"  Classes: {train_metadata.thing_classes}")
print(f"  Training images: {len(train_dataset)}")
print(f"  Validation images: {len(val_dataset)}")

# %% [markdown]
# ## Cell 7: Conservative Augmentation Mapper

# %%
def get_conservative_augmentation():
    """Conservative augmentation pipeline for histopathology."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=10, p=0.3, border_mode=cv2.BORDER_REFLECT),
        
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=15, p=1.0),
        ], p=0.4),
        
        A.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.03, p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.15),
        A.GaussNoise(std_range=(0.01, 0.02), p=0.1),
        
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['category_ids'],
        min_area=1,
        min_visibility=0.3,
    ))


class BalancedAugMapper:
    """DatasetMapper with conservative augmentations for 5-class training."""
    
    def __init__(self, cfg, is_train=True):
        self.cfg = cfg
        self.is_train = is_train
        self.augmentation = get_conservative_augmentation() if is_train else None
        self.image_format = cfg.INPUT.FORMAT
        
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        annotations = dataset_dict.get("annotations", [])
        
        if self.is_train and self.augmentation and len(annotations) > 0:
            bboxes, category_ids, masks = [], [], []
            
            for ann in annotations:
                bbox = ann["bbox"]
                if ann.get("bbox_mode") == BoxMode.XYXY_ABS:
                    bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                bboxes.append(bbox)
                category_ids.append(ann["category_id"])
                
                h, w = image.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                for poly in ann.get("segmentation", []):
                    pts = np.array(poly).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
                masks.append(mask)
            
            try:
                if self.image_format == "BGR":
                    image_rgb = image[:, :, ::-1]
                else:
                    image_rgb = image
                
                transformed = self.augmentation(
                    image=image_rgb, bboxes=bboxes,
                    category_ids=category_ids, masks=masks
                )
                
                image_aug = transformed["image"]
                if self.image_format == "BGR":
                    image_aug = image_aug[:, :, ::-1]
                image = image_aug
                
                new_annotations = []
                for bbox, cat_id, mask_aug in zip(
                    transformed["bboxes"], 
                    transformed["category_ids"], 
                    transformed["masks"]
                ):
                    contours, _ = cv2.findContours(
                        mask_aug.astype(np.uint8), 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    segmentation = [c.flatten().tolist() for c in contours if c.size >= 6]
                    
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
                pass
        
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        
        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict
        
        annos = [utils.transform_instance_annotations(obj, [], image.shape[:2]) for obj in annotations]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        
        return dataset_dict

# %% [markdown]
# ## Cell 8: Custom Trainer with RepeatFactorSampler

# %%
class BalancedTrainer(DefaultTrainer):
    """
    Trainer with:
    1. RepeatFactorTrainingSampler for class balancing
    2. Conservative augmentations
    """
    
    @classmethod
    def build_train_loader(cls, cfg):
        """Build train loader with repeat factor sampling."""
        mapper = BalancedAugMapper(cfg, is_train=True)
        
        # Get dataset
        dataset = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        
        # Calculate repeat factors for each dataset entry
        # RepeatFactorTrainingSampler expects a list of repeat factors
        repeat_factors = []
        
        for d in dataset:
            # Get categories in this image
            categories = set(ann['category_id'] for ann in d.get('annotations', []))
            
            if categories:
                # Use max repeat factor of all categories in image
                max_repeat = max(category_repeat.get(cat_id, 1.0) for cat_id in categories)
            else:
                max_repeat = 1.0
            
            repeat_factors.append(max_repeat)
        
        repeat_factors = torch.tensor(repeat_factors)
        
        # Create sampler
        sampler = RepeatFactorTrainingSampler(repeat_factors)
        
        return build_detection_train_loader(cfg, mapper=mapper, sampler=sampler)

# %% [markdown]
# ## Cell 9: Model Configuration

# %%
cfg = get_cfg()

# Model
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)

# Dataset
cfg.DATASETS.TRAIN = ("tumor_bud_5class_train",)
cfg.DATASETS.TEST = ("tumor_bud_5class_val",)
cfg.DATALOADER.NUM_WORKERS = 2

# Output
cfg.OUTPUT_DIR = OUTPUT_DIR
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Solver
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.002
cfg.SOLVER.MAX_ITER = 10000  # More iterations for 5-class
cfg.SOLVER.STEPS = (6000, 8000)
cfg.SOLVER.GAMMA = 0.5
cfg.SOLVER.WARMUP_ITERS = 500

# Model - 5 classes
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

# Small object optimization
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
cfg.MODEL.RPN.NMS_THRESH = 0.7
cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.4]

# Evaluation
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.35
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4

print("Configuration Summary:")
print(f"  Model: R50-FPN (5-class)")
print(f"  Classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
print(f"  Max iterations: {cfg.SOLVER.MAX_ITER}")
print(f"  With RepeatFactorSampler for class balancing")

# %% [markdown]
# ## Cell 10: Train Model

# %%
print("=" * 60)
print("STARTING 5-CLASS TRAINING WITH CLASS BALANCING")
print("=" * 60)
print(f"Classes: {train_metadata.thing_classes}")
print(f"With RepeatFactorTrainingSampler for rare classes")
print("=" * 60)

trainer = BalancedTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

print("\nTraining complete!")
print(f"Model saved to: {cfg.OUTPUT_DIR}")

# %% [markdown]
# ## Cell 11: Evaluation

# %%
# Load trained model
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
predictor = DefaultPredictor(cfg)

def evaluate_multiclass(dataset, predictor, class_names, iou_threshold=0.3):
    """Evaluate 5-class detection with per-class metrics."""
    print(f"\n{'='*70}")
    print("5-CLASS TUMOR BUD DETECTION EVALUATION")
    print(f"{'='*70}")
    
    num_classes = len(class_names)
    
    # Per-class metrics
    tp_per_class = {i: 0 for i in range(num_classes)}
    fp_per_class = {i: 0 for i in range(num_classes)}
    fn_per_class = {i: 0 for i in range(num_classes)}
    gt_per_class = {i: 0 for i in range(num_classes)}
    
    for d in dataset:
        gt_annotations = d["annotations"]
        gt_boxes = []
        gt_classes = []
        
        for ann in gt_annotations:
            bbox = ann["bbox"]
            gt_boxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
            gt_classes.append(ann["category_id"] - 1)  # 0-indexed
        
        img = cv2.imread(d["file_name"])
        if img is None:
            continue
        
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        
        pred_boxes = instances.pred_boxes.tensor.numpy().tolist() if len(instances) > 0 else []
        pred_classes = instances.pred_classes.numpy().tolist() if len(instances) > 0 else []
        
        # Count GT per class
        for cls in gt_classes:
            gt_per_class[cls] += 1
        
        # Match predictions to GT (same class only)
        matched_gt = set()
        
        for pred_idx, (pred_box, pred_cls) in enumerate(zip(pred_boxes, pred_classes)):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                if gt_cls != pred_cls or gt_idx in matched_gt:
                    continue
                
                # Calculate IoU
                x1 = max(pred_box[0], gt_box[0])
                y1 = max(pred_box[1], gt_box[1])
                x2 = min(pred_box[2], gt_box[2])
                y2 = min(pred_box[3], gt_box[3])
                inter = max(0, x2-x1) * max(0, y2-y1)
                area1 = (pred_box[2]-pred_box[0]) * (pred_box[3]-pred_box[1])
                area2 = (gt_box[2]-gt_box[0]) * (gt_box[3]-gt_box[1])
                iou = inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp_per_class[pred_cls] += 1
                matched_gt.add(best_gt_idx)
            else:
                fp_per_class[pred_cls] += 1
        
        # Count FN
        for gt_idx, gt_cls in enumerate(gt_classes):
            if gt_idx not in matched_gt:
                fn_per_class[gt_cls] += 1
    
    # Print per-class results
    print(f"\n{'Class':<15} | {'GT':>6} | {'TP':>6} | {'FP':>6} | {'FN':>6} | {'Prec':>8} | {'Recall':>8} | {'F1':>8}")
    print("-" * 75)
    
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for i, name in enumerate(class_names):
        tp = tp_per_class[i]
        fp = fp_per_class[i]
        fn = fn_per_class[i]
        gt = gt_per_class[i]
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        print(f"{name:<15} | {gt:>6} | {tp:>6} | {fp:>6} | {fn:>6} | {prec:>8.3f} | {rec:>8.3f} | {f1:>8.3f}")
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Overall
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_prec * overall_rec / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0
    
    print("-" * 75)
    total_gt = sum(gt_per_class.values())
    print(f"{'OVERALL':<15} | {total_gt:>6} | {total_tp:>6} | {total_fp:>6} | {total_fn:>6} | {overall_prec:>8.3f} | {overall_rec:>8.3f} | {overall_f1:>8.3f}")
    
    return {
        "precision": overall_prec,
        "recall": overall_rec,
        "f1": overall_f1,
        "tp_per_class": dict(tp_per_class),
        "fp_per_class": dict(fp_per_class),
        "fn_per_class": dict(fn_per_class),
    }

# Run evaluation
results = evaluate_multiclass(val_dataset, predictor, train_metadata.thing_classes)

# Save results
with open(os.path.join(cfg.OUTPUT_DIR, "evaluation_results.json"), 'w') as f:
    json.dump(results, f, indent=2)

# %% [markdown]
# ## Cell 12: Summary

# %%
print("\n" + "=" * 60)
print("5-CLASS TRAINING COMPLETE")
print("=" * 60)
print(f"\nOverall Metrics:")
print(f"  Precision: {results['precision']*100:.1f}%")
print(f"  Recall:    {results['recall']*100:.1f}%")
print(f"  F1-Score:  {results['f1']*100:.1f}%")
print(f"\nModel saved to: {cfg.OUTPUT_DIR}")
