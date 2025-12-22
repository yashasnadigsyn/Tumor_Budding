# ============================================================================
# HYPERPARAMETER TUNING FOR TUMOR BUD DETECTION
# ============================================================================
# This script provides systematic hyperparameter tuning for Mask R-CNN.
# 
# Key hyperparameters to tune:
# 1. Anchor sizes (for small tumor buds)
# 2. Learning rate and schedule
# 3. NMS thresholds
# 4. IoU thresholds
# 5. Score threshold (for recall vs precision trade-off)
# 
# Run each configuration and compare results to find optimal settings.
# ============================================================================

# %% [markdown]
# ## Cell 1: Setup

# %%
# Uncomment for Google Colab
# from google.colab import drive
# drive.mount('/content/drive', force_remount=False)
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
import time
import itertools
from pathlib import Path
from collections import Counter
from matplotlib import pyplot as plt
from typing import Dict, List, Any

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode

import albumentations as A

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## Cell 3: Configuration

# %%
BASE_DIR = Path(".")

# Use binary dataset (single class - recommended)
TRAIN_JSON = str(BASE_DIR / "dataset_binary" / "train" / "annotations.json")
TRAIN_IMAGES = str(BASE_DIR / "dataset_binary" / "train")
VAL_JSON = str(BASE_DIR / "dataset_binary" / "test" / "annotations.json")
VAL_IMAGES = str(BASE_DIR / "dataset_binary" / "test")
NUM_CLASSES = 1

OUTPUT_BASE = str(BASE_DIR / "models" / "hyperparameter_tuning")
os.makedirs(OUTPUT_BASE, exist_ok=True)

# %% [markdown]
# ## Cell 4: Define Hyperparameter Search Space

# %%
# =============================================================================
# HYPERPARAMETER CONFIGURATIONS TO TEST
# =============================================================================
# Each experiment tests one hyperparameter change from baseline

HYPERPARAMETER_EXPERIMENTS = {
    # --- BASELINE ---
    "baseline": {
        "description": "Default configuration",
        "anchor_sizes": [[16], [32], [64], [128], [256]],
        "base_lr": 0.001,
        "max_iter": 5000,
        "nms_thresh_train": 0.7,
        "nms_thresh_test": 0.3,
        "score_thresh_test": 0.4,
        "iou_thresh_train": 0.5,
    },
    
    # --- ANCHOR SIZES (for small objects) ---
    "anchors_smaller": {
        "description": "Smaller anchors: [8,16,32,64,128] for tiny tumor buds",
        "anchor_sizes": [[8], [16], [32], [64], [128]],
        "base_lr": 0.001,
        "max_iter": 5000,
        "nms_thresh_train": 0.7,
        "nms_thresh_test": 0.3,
        "score_thresh_test": 0.4,
        "iou_thresh_train": 0.5,
    },
    
    "anchors_very_small": {
        "description": "Very small anchors: [4,8,16,32,64] for very tiny buds",
        "anchor_sizes": [[4], [8], [16], [32], [64]],
        "base_lr": 0.001,
        "max_iter": 5000,
        "nms_thresh_train": 0.7,
        "nms_thresh_test": 0.3,
        "score_thresh_test": 0.4,
        "iou_thresh_train": 0.5,
    },
    
    # --- LEARNING RATE ---
    "lr_higher": {
        "description": "Higher learning rate: 0.0025",
        "anchor_sizes": [[8], [16], [32], [64], [128]],
        "base_lr": 0.0025,
        "max_iter": 5000,
        "nms_thresh_train": 0.7,
        "nms_thresh_test": 0.3,
        "score_thresh_test": 0.4,
        "iou_thresh_train": 0.5,
    },
    
    "lr_lower": {
        "description": "Lower learning rate: 0.0005",
        "anchor_sizes": [[8], [16], [32], [64], [128]],
        "base_lr": 0.0005,
        "max_iter": 8000,  # More iterations for lower LR
        "nms_thresh_train": 0.7,
        "nms_thresh_test": 0.3,
        "score_thresh_test": 0.4,
        "iou_thresh_train": 0.5,
    },
    
    # --- MORE TRAINING ---
    "longer_training": {
        "description": "Longer training: 10000 iterations",
        "anchor_sizes": [[8], [16], [32], [64], [128]],
        "base_lr": 0.002,
        "max_iter": 10000,
        "nms_thresh_train": 0.7,
        "nms_thresh_test": 0.3,
        "score_thresh_test": 0.4,
        "iou_thresh_train": 0.5,
    },
    
    # --- NMS THRESHOLDS ---
    "nms_higher": {
        "description": "Higher NMS (0.5): keep more overlapping predictions",
        "anchor_sizes": [[8], [16], [32], [64], [128]],
        "base_lr": 0.001,
        "max_iter": 5000,
        "nms_thresh_train": 0.7,
        "nms_thresh_test": 0.5,
        "score_thresh_test": 0.4,
        "iou_thresh_train": 0.5,
    },
    
    "nms_lower": {
        "description": "Lower NMS (0.2): more strict, fewer overlaps",
        "anchor_sizes": [[8], [16], [32], [64], [128]],
        "base_lr": 0.001,
        "max_iter": 5000,
        "nms_thresh_train": 0.7,
        "nms_thresh_test": 0.2,
        "score_thresh_test": 0.4,
        "iou_thresh_train": 0.5,
    },
    
    # --- SCORE THRESHOLD (Recall vs Precision) ---
    "high_recall": {
        "description": "Low score threshold (0.25) for HIGH RECALL (medical priority)",
        "anchor_sizes": [[8], [16], [32], [64], [128]],
        "base_lr": 0.001,
        "max_iter": 5000,
        "nms_thresh_train": 0.7,
        "nms_thresh_test": 0.4,
        "score_thresh_test": 0.25,  # Lower = higher recall
        "iou_thresh_train": 0.5,
    },
    
    "high_precision": {
        "description": "High score threshold (0.5) for HIGH PRECISION",
        "anchor_sizes": [[8], [16], [32], [64], [128]],
        "base_lr": 0.001,
        "max_iter": 5000,
        "nms_thresh_train": 0.7,
        "nms_thresh_test": 0.3,
        "score_thresh_test": 0.5,  # Higher = higher precision
        "iou_thresh_train": 0.5,
    },
    
    # --- IoU THRESHOLD FOR TRAINING ---
    "iou_train_lower": {
        "description": "Lower IoU threshold (0.4) for training - catch more positives",
        "anchor_sizes": [[8], [16], [32], [64], [128]],
        "base_lr": 0.001,
        "max_iter": 5000,
        "nms_thresh_train": 0.7,
        "nms_thresh_test": 0.3,
        "score_thresh_test": 0.4,
        "iou_thresh_train": 0.4,
    },
    
    # --- OPTIMIZED CONFIGURATION (based on research) ---
    "optimized": {
        "description": "Optimized for tumor buds: small anchors, high recall, longer training",
        "anchor_sizes": [[8], [16], [32], [64], [128]],
        "base_lr": 0.002,
        "max_iter": 8000,
        "nms_thresh_train": 0.7,
        "nms_thresh_test": 0.4,
        "score_thresh_test": 0.3,  # Prioritize recall
        "iou_thresh_train": 0.4,
    },
}

print("Hyperparameter Experiments:")
print("-" * 60)
for name, config in HYPERPARAMETER_EXPERIMENTS.items():
    print(f"  {name}: {config['description']}")
print("-" * 60)

# %% [markdown]
# ## Cell 5: Select Experiment

# %%
# =============================================================================
# SELECT WHICH EXPERIMENT TO RUN
# =============================================================================

SELECTED_EXPERIMENT = "optimized"  # Change this to run different experiments

print(f"\nSelected: {SELECTED_EXPERIMENT}")
print(f"Description: {HYPERPARAMETER_EXPERIMENTS[SELECTED_EXPERIMENT]['description']}")

# %% [markdown]
# ## Cell 6: Register Dataset

# %%
dataset_name = "tumor_bud_hp_train"
val_dataset_name = "tumor_bud_hp_val"

if dataset_name in DatasetCatalog.list():
    DatasetCatalog.remove(dataset_name)
    MetadataCatalog.remove(dataset_name)
if val_dataset_name in DatasetCatalog.list():
    DatasetCatalog.remove(val_dataset_name)
    MetadataCatalog.remove(val_dataset_name)

register_coco_instances(dataset_name, {}, TRAIN_JSON, TRAIN_IMAGES)
register_coco_instances(val_dataset_name, {}, VAL_JSON, VAL_IMAGES)

train_metadata = MetadataCatalog.get(dataset_name)
train_dataset = DatasetCatalog.get(dataset_name)
val_dataset = DatasetCatalog.get(val_dataset_name)

print(f"Training: {len(train_dataset)} images")
print(f"Validation: {len(val_dataset)} images")

# %% [markdown]
# ## Cell 7: Augmentation and Trainer

# %%
def get_conservative_augmentation():
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
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], min_area=1, min_visibility=0.3))


class HPTuningMapper:
    def __init__(self, cfg, is_train=True):
        self.cfg, self.is_train = cfg, is_train
        self.augmentation = get_conservative_augmentation() if is_train else None
        self.image_format = cfg.INPUT.FORMAT
        
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        annotations = dataset_dict.get("annotations", [])
        
        if self.is_train and self.augmentation and annotations:
            bboxes, category_ids, masks = [], [], []
            for ann in annotations:
                bbox = ann["bbox"]
                if ann.get("bbox_mode") == BoxMode.XYXY_ABS:
                    bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
                bboxes.append(bbox)
                category_ids.append(ann["category_id"])
                h, w = image.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                for poly in ann.get("segmentation", []):
                    pts = np.array(poly).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
                masks.append(mask)
            
            try:
                image_rgb = image[:, :, ::-1] if self.image_format == "BGR" else image
                t = self.augmentation(image=image_rgb, bboxes=bboxes, category_ids=category_ids, masks=masks)
                image = t["image"][:, :, ::-1] if self.image_format == "BGR" else t["image"]
                annotations = []
                for bbox, cat_id, mask_aug in zip(t["bboxes"], t["category_ids"], t["masks"]):
                    contours, _ = cv2.findContours(mask_aug.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    segmentation = [c.flatten().tolist() for c in contours if c.size >= 6]
                    if segmentation:
                        annotations.append({"bbox": list(bbox), "bbox_mode": BoxMode.XYWH_ABS, "category_id": cat_id, "segmentation": segmentation, "iscrowd": 0})
            except:
                pass
        
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict
        annos = [utils.transform_instance_annotations(obj, [], image.shape[:2]) for obj in annotations]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict


class HPTuningTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=HPTuningMapper(cfg, is_train=True))

# %% [markdown]
# ## Cell 8: Build Configuration

# %%
def build_hp_config(experiment_name: str) -> Any:
    """Build configuration for a hyperparameter experiment."""
    hp = HYPERPARAMETER_EXPERIMENTS[experiment_name]
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = 2
    
    cfg.OUTPUT_DIR = os.path.join(OUTPUT_BASE, experiment_name)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Apply hyperparameters
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = hp["anchor_sizes"]
    cfg.SOLVER.BASE_LR = hp["base_lr"]
    cfg.SOLVER.MAX_ITER = hp["max_iter"]
    cfg.SOLVER.STEPS = (int(hp["max_iter"]*0.6), int(hp["max_iter"]*0.85))
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.IMS_PER_BATCH = 4
    
    cfg.MODEL.RPN.NMS_THRESH = hp["nms_thresh_train"]
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = hp["nms_thresh_test"]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = hp["score_thresh_test"]
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [hp["iou_thresh_train"]]
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    
    # Additional optimizations
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    
    return cfg

cfg = build_hp_config(SELECTED_EXPERIMENT)
hp = HYPERPARAMETER_EXPERIMENTS[SELECTED_EXPERIMENT]

print(f"\nConfiguration for {SELECTED_EXPERIMENT}:")
print(f"  Anchor sizes: {hp['anchor_sizes']}")
print(f"  Learning rate: {hp['base_lr']}")
print(f"  Max iterations: {hp['max_iter']}")
print(f"  NMS threshold (test): {hp['nms_thresh_test']}")
print(f"  Score threshold: {hp['score_thresh_test']}")
print(f"  IoU threshold (train): {hp['iou_thresh_train']}")

# %% [markdown]
# ## Cell 9: Train

# %%
print("=" * 60)
print(f"TRAINING: {SELECTED_EXPERIMENT}")
print(f"Description: {hp['description']}")
print("=" * 60)

start_time = time.time()
trainer = HPTuningTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
training_time = time.time() - start_time

print(f"\nTraining completed in {training_time/60:.1f} minutes")

# %% [markdown]
# ## Cell 10: Evaluate

# %%
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
predictor = DefaultPredictor(cfg)

def evaluate_hp(dataset, predictor, iou_threshold=0.3):
    """Evaluate with multiple IoU thresholds."""
    total_tp, total_fp, total_fn = 0, 0, 0
    dice_scores, count_errors = [], []
    
    for d in dataset:
        gt_boxes = [[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in [ann["bbox"] for ann in d["annotations"]]]
        img = cv2.imread(d["file_name"])
        if img is None:
            continue
        
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        pred_boxes = instances.pred_boxes.tensor.numpy().tolist() if len(instances) > 0 else []
        
        matched_gt = set()
        for pred_box in pred_boxes:
            best_iou, best_idx = 0, -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                x1, y1 = max(pred_box[0], gt_box[0]), max(pred_box[1], gt_box[1])
                x2, y2 = min(pred_box[2], gt_box[2]), min(pred_box[3], gt_box[3])
                inter = max(0, x2-x1) * max(0, y2-y1)
                union = (pred_box[2]-pred_box[0])*(pred_box[3]-pred_box[1]) + (gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1]) - inter
                iou = inter/union if union > 0 else 0
                if iou > best_iou:
                    best_iou, best_idx = iou, gt_idx
            if best_iou >= iou_threshold:
                matched_gt.add(best_idx)
                total_tp += 1
            else:
                total_fp += 1
        total_fn += len(gt_boxes) - len(matched_gt)
        
        # DICE
        gt_count, pred_count = len(gt_boxes), len(pred_boxes)
        dice = (2 * min(gt_count, pred_count)) / (gt_count + pred_count) if (gt_count + pred_count) > 0 else 1.0
        dice_scores.append(dice)
        count_errors.append(abs(gt_count - pred_count))
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "dice": np.mean(dice_scores), "mae": np.mean(count_errors),
        "tp": total_tp, "fp": total_fp, "fn": total_fn
    }

results = evaluate_hp(val_dataset, predictor)

print(f"\n{'='*60}")
print(f"RESULTS: {SELECTED_EXPERIMENT}")
print(f"{'='*60}")
print(f"  Precision: {results['precision']*100:.1f}%")
print(f"  Recall:    {results['recall']*100:.1f}%")
print(f"  F1-Score:  {results['f1']*100:.1f}%")
print(f"  DICE:      {results['dice']*100:.1f}%")
print(f"  MAE:       {results['mae']:.2f}")
print(f"  Training:  {training_time/60:.1f} min")

# Save results
results["experiment"] = SELECTED_EXPERIMENT
results["hyperparameters"] = hp
results["training_time_seconds"] = training_time
with open(os.path.join(cfg.OUTPUT_DIR, "results.json"), 'w') as f:
    json.dump(results, f, indent=2, default=str)

# %% [markdown]
# ## Cell 11: Compare All Experiments

# %%
def compare_all_experiments():
    """Load and compare all hyperparameter experiment results."""
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING COMPARISON")
    print("=" * 80)
    print(f"{'Experiment':<20} | {'Prec':>8} | {'Recall':>8} | {'F1':>8} | {'DICE':>8} | {'MAE':>6}")
    print("-" * 80)
    
    all_results = []
    for exp_name in HYPERPARAMETER_EXPERIMENTS.keys():
        result_path = os.path.join(OUTPUT_BASE, exp_name, "results.json")
        if Path(result_path).exists():
            with open(result_path, 'r') as f:
                r = json.load(f)
            print(f"{exp_name:<20} | {r['precision']*100:>7.1f}% | {r['recall']*100:>7.1f}% | {r['f1']*100:>7.1f}% | {r['dice']*100:>7.1f}% | {r['mae']:>6.2f}")
            all_results.append((exp_name, r))
    
    print("=" * 80)
    
    if all_results:
        best_recall = max(all_results, key=lambda x: x[1]['recall'])
        best_f1 = max(all_results, key=lambda x: x[1]['f1'])
        best_dice = max(all_results, key=lambda x: x[1]['dice'])
        
        print(f"\nBest Recall: {best_recall[0]} ({best_recall[1]['recall']*100:.1f}%)")
        print(f"Best F1:     {best_f1[0]} ({best_f1[1]['f1']*100:.1f}%)")
        print(f"Best DICE:   {best_dice[0]} ({best_dice[1]['dice']*100:.1f}%)")
        
        print("\n--- RECOMMENDATION ---")
        print(f"For medical imaging (prioritize recall): Use '{best_recall[0]}'")

compare_all_experiments()

# %% [markdown]
# ## Cell 12: Best Configuration Summary

# %%
print("""
================================================================================
HYPERPARAMETER TUNING SUMMARY
================================================================================

RECOMMENDED SETTINGS FOR TUMOR BUD DETECTION:

1. ANCHOR SIZES: [[8], [16], [32], [64], [128]]
   - Smaller anchors help detect small tumor buds
   - Default [[32], [64], ...] misses small objects

2. LEARNING RATE: 0.002 with warmup
   - Stable training with good convergence
   - Use warmup_iters = 500

3. TRAINING ITERATIONS: 8000-10000
   - More iterations = better convergence
   - Watch for overfitting

4. SCORE THRESHOLD: 0.25-0.35
   - LOWER for higher RECALL (medical priority)
   - HIGHER for higher PRECISION

5. NMS THRESHOLD: 0.3-0.4
   - Balance between avoiding duplicates and catching overlapping buds

6. IoU THRESHOLD (TRAINING): 0.4
   - Slightly lower than default 0.5
   - Helps with irregular tumor bud shapes

QUICK SETTINGS FOR DIFFERENT PRIORITIES:
- HIGH RECALL (minimize missed buds):  score_thresh=0.25, iou_train=0.4
- HIGH PRECISION (minimize false positives): score_thresh=0.5, iou_train=0.5
- BALANCED: score_thresh=0.35, iou_train=0.4
================================================================================
""")
