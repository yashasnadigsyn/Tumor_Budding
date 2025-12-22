# ============================================================================
# ALTERNATIVE MODEL ARCHITECTURES COMPARISON
# ============================================================================
# This script trains and compares different Detectron2 model architectures
# for tumor bud detection:
# 
# 1. R50-FPN (current baseline)
# 2. R101-FPN (deeper backbone, better features)
# 3. X101-32x8d-FPN (best accuracy, but heavier)
# 4. Cascade R-CNN (multi-stage refinement)
# 
# Each model is trained with the same hyperparameters for fair comparison.
# ============================================================================

# %% [markdown]
# ## Cell 1: Setup (Google Colab)

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
from pathlib import Path
from collections import Counter
from matplotlib import pyplot as plt

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode

import albumentations as A

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU Memory: {gpu_mem:.1f} GB")

# %% [markdown]
# ## Cell 3: Configuration

# %%
# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(".")

# Use binary dataset (recommended) or 5-class
USE_BINARY = True

if USE_BINARY:
    TRAIN_JSON = str(BASE_DIR / "dataset_binary" / "train" / "annotations.json")
    TRAIN_IMAGES = str(BASE_DIR / "dataset_binary" / "train")
    VAL_JSON = str(BASE_DIR / "dataset_binary" / "test" / "annotations.json")
    VAL_IMAGES = str(BASE_DIR / "dataset_binary" / "test")
    NUM_CLASSES = 1
else:
    TRAIN_JSON = str(BASE_DIR / "dataset" / "train" / "annotations.json")
    TRAIN_IMAGES = str(BASE_DIR / "dataset" / "train")
    VAL_JSON = str(BASE_DIR / "dataset" / "test" / "annotations.json")
    VAL_IMAGES = str(BASE_DIR / "dataset" / "test")
    NUM_CLASSES = 5

OUTPUT_BASE = str(BASE_DIR / "models")

# Verify paths
for path in [TRAIN_JSON, VAL_JSON]:
    if Path(path).exists():
        print(f"✓ {path}")
    else:
        print(f"✗ Missing: {path}")

# %% [markdown]
# ## Cell 4: Define Model Configurations

# %%
# =============================================================================
# MODEL CONFIGURATIONS TO COMPARE
# =============================================================================
# Each entry: (name, config_file, checkpoint_url, batch_size_hint)
# batch_size_hint is suggested based on 16GB GPU memory

MODEL_CONFIGS = {
    "R50-FPN": {
        "config": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "batch_size": 4,  # Works well on 16GB
        "description": "Baseline - ResNet-50 with FPN",
    },
    "R101-FPN": {
        "config": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        "batch_size": 4,  # May need 3 if OOM
        "description": "Deeper backbone - ResNet-101 with FPN",
    },
    "X101-FPN": {
        "config": "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
        "batch_size": 2,  # Heavy model, reduce batch size
        "description": "Best accuracy - ResNeXt-101 with FPN",
    },
    "Cascade-R50": {
        "config": "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml",
        "batch_size": 3,  # Multi-stage needs more memory
        "description": "Multi-stage refinement - Cascade R-CNN",
    },
}

print("Available Model Configurations:")
print("-" * 60)
for name, config in MODEL_CONFIGS.items():
    print(f"  {name:<15} - {config['description']}")
    print(f"                   Batch size: {config['batch_size']}")
print("-" * 60)

# %% [markdown]
# ## Cell 5: Select Model to Train

# %%
# =============================================================================
# SELECT WHICH MODEL TO TRAIN
# =============================================================================
# Change this to train different models

SELECTED_MODEL = "R101-FPN"  # Options: "R50-FPN", "R101-FPN", "X101-FPN", "Cascade-R50"

print(f"\nSelected model: {SELECTED_MODEL}")
print(f"Description: {MODEL_CONFIGS[SELECTED_MODEL]['description']}")

# %% [markdown]
# ## Cell 6: Register Dataset

# %%
# Register datasets
dataset_name = "tumor_bud_arch_train"
val_dataset_name = "tumor_bud_arch_val"

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

print(f"\nDataset:")
print(f"  Classes: {train_metadata.thing_classes}")
print(f"  Training: {len(train_dataset)} images")
print(f"  Validation: {len(val_dataset)} images")

# %% [markdown]
# ## Cell 7: Conservative Augmentation

# %%
def get_conservative_augmentation():
    """Histopathology-optimized augmentations."""
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


class ModelComparisonMapper:
    """DatasetMapper for model comparison experiments."""
    
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
                transformed = self.augmentation(
                    image=image_rgb, bboxes=bboxes, category_ids=category_ids, masks=masks
                )
                image = transformed["image"][:, :, ::-1] if self.image_format == "BGR" else transformed["image"]
                
                annotations = []
                for bbox, cat_id, mask_aug in zip(transformed["bboxes"], transformed["category_ids"], transformed["masks"]):
                    contours, _ = cv2.findContours(mask_aug.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    segmentation = [c.flatten().tolist() for c in contours if c.size >= 6]
                    if segmentation:
                        annotations.append({
                            "bbox": list(bbox), "bbox_mode": BoxMode.XYWH_ABS,
                            "category_id": cat_id, "segmentation": segmentation, "iscrowd": 0
                        })
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


class ModelComparisonTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=ModelComparisonMapper(cfg, is_train=True))

# %% [markdown]
# ## Cell 8: Build Configuration for Selected Model

# %%
def build_config(model_name, output_dir):
    """Build configuration for a specific model."""
    model_config = MODEL_CONFIGS[model_name]
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_config["config"]))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config["config"])
    
    # Dataset
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = 2
    
    # Output
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Solver - consistent across models for fair comparison
    cfg.SOLVER.IMS_PER_BATCH = model_config["batch_size"]
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 8000
    cfg.SOLVER.STEPS = (5000, 7000)
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.WARMUP_ITERS = 500
    
    # Model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    
    # Small object optimization
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.NMS_THRESH = 0.7
    
    # Evaluation
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
    
    return cfg

# Build config for selected model
output_dir = os.path.join(OUTPUT_BASE, f"architecture_{SELECTED_MODEL}")
cfg = build_config(SELECTED_MODEL, output_dir)

print(f"\nConfiguration for {SELECTED_MODEL}:")
print(f"  Config file: {MODEL_CONFIGS[SELECTED_MODEL]['config']}")
print(f"  Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
print(f"  Classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
print(f"  Output: {cfg.OUTPUT_DIR}")

# %% [markdown]
# ## Cell 9: Train Selected Model

# %%
print("=" * 60)
print(f"TRAINING: {SELECTED_MODEL}")
print(f"Description: {MODEL_CONFIGS[SELECTED_MODEL]['description']}")
print("=" * 60)

start_time = time.time()

trainer = ModelComparisonTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time/60:.1f} minutes")

# %% [markdown]
# ## Cell 10: Evaluate Model

# %%
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
predictor = DefaultPredictor(cfg)

def evaluate_model(dataset, predictor, iou_threshold=0.3):
    """Evaluate detection performance."""
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for d in dataset:
        gt_boxes = []
        for ann in d["annotations"]:
            bbox = ann["bbox"]
            gt_boxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
        
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
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1, "tp": total_tp, "fp": total_fp, "fn": total_fn}

results = evaluate_model(val_dataset, predictor)

print(f"\n{'='*60}")
print(f"RESULTS: {SELECTED_MODEL}")
print(f"{'='*60}")
print(f"  Precision: {results['precision']*100:.1f}%")
print(f"  Recall:    {results['recall']*100:.1f}%")
print(f"  F1-Score:  {results['f1']*100:.1f}%")
print(f"  Training time: {training_time/60:.1f} min")

# Save results
results["model"] = SELECTED_MODEL
results["training_time_seconds"] = training_time
with open(os.path.join(cfg.OUTPUT_DIR, "results.json"), 'w') as f:
    json.dump(results, f, indent=2)

# %% [markdown]
# ## Cell 11: Compare All Models (Run after training each)

# %%
def compare_all_models():
    """Load and compare results from all trained models."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<15} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'Time (min)':>10}")
    print("-" * 70)
    
    all_results = []
    for model_name in MODEL_CONFIGS.keys():
        result_path = os.path.join(OUTPUT_BASE, f"architecture_{model_name}", "results.json")
        if Path(result_path).exists():
            with open(result_path, 'r') as f:
                r = json.load(f)
            print(f"{model_name:<15} | {r['precision']*100:>9.1f}% | {r['recall']*100:>9.1f}% | {r['f1']*100:>9.1f}% | {r.get('training_time_seconds', 0)/60:>10.1f}")
            all_results.append(r)
        else:
            print(f"{model_name:<15} | {'Not trained':>10} |")
    
    print("=" * 70)
    
    if all_results:
        best_f1 = max(all_results, key=lambda x: x['f1'])
        best_recall = max(all_results, key=lambda x: x['recall'])
        print(f"\nBest F1: {best_f1['model']} ({best_f1['f1']*100:.1f}%)")
        print(f"Best Recall: {best_recall['model']} ({best_recall['recall']*100:.1f}%)")

# Run comparison
compare_all_models()

# %% [markdown]
# ## Cell 12: Recommendations

# %%
print("""
================================================================================
RECOMMENDATIONS FOR MODEL SELECTION
================================================================================

1. R50-FPN (Baseline)
   - Fast training, good for iteration
   - Use for quick experiments
   
2. R101-FPN (Recommended for your case)
   - Better feature extraction than R50
   - Good balance of accuracy and speed
   - Works well on 16GB GPU
   
3. X101-FPN (Best accuracy)
   - Highest accuracy, but slower
   - Use if training time is not a concern
   - May need batch size = 2 on 16GB GPU
   
4. Cascade R-CNN (For difficult cases)
   - Multi-stage refinement
   - Better for objects with varying sizes
   - Good for reducing false positives

For medical imaging where RECALL is priority:
- Start with R101-FPN
- If recall < 80%, try X101-FPN
- Lower SCORE_THRESH_TEST to 0.25 for higher recall
================================================================================
""")
