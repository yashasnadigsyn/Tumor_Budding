# ## PreRequisite
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

# ## Install Detectron2
!python -m pip install pyyaml==5.1
import sys, os, distutils.core
# Note: This is a faster way to install detectron2 in Colab, but it does not include all functionalities (e.g. compiled operators).
# See https://detectron2.readthedocs.io/tutorials/install.html for full installation instructions
!git clone 'https://github.com/facebookresearch/detectron2'
dist = distutils.core.run_setup("./detectron2/setup.py")
!python -m pip install {' '.join([f"'{x}'" for x in dist.install_requires])}
sys.path.insert(0, os.path.abspath('./detectron2'))

# Properly install detectron2. (Please do not install twice in both ways)
# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

import torch, detectron2
!nvcc --version
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

# ## Imports
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
from google.colab.patches import cv2_imshow
from matplotlib import pyplot as plt

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode
import copy

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ## Dataset
TRAIN_JSON = "/content/drive/MyDrive/dataset/train/annotations.json"
TRAIN_IMAGES = "/content/drive/MyDrive/dataset/train"
VAL_JSON = "/content/drive/MyDrive/dataset/test/annotations.json"
VAL_IMAGES = "/content/drive/MyDrive/dataset/test/"

if "my_dataset_train" not in DatasetCatalog.list():
    register_coco_instances("my_dataset_train", {}, TRAIN_JSON, TRAIN_IMAGES)
if "my_dataset_validation" not in DatasetCatalog.list():
    register_coco_instances("my_dataset_validation", {}, VAL_JSON, VAL_IMAGES)

train_metadata = MetadataCatalog.get("my_dataset_train")
train_dataset_dicts = DatasetCatalog.get("my_dataset_train")
val_metadata = MetadataCatalog.get("my_dataset_validation")
val_dataset_dicts = DatasetCatalog.get("my_dataset_validation")

print(f"Classes: {train_metadata.thing_classes}")
print(f"Training images: {len(train_dataset_dicts)}")
print(f"Validation images: {len(val_dataset_dicts)}")

def visualize_random_samples(dataset_dicts, metadata, num_samples=2):
    """Visualize random samples from the dataset."""
    for d in random.sample(dataset_dicts, min(num_samples, len(dataset_dicts))):
        img = cv2.imread(d["file_name"])
        if img is None:
            print(f"Warning: Could not load image from {d['file_name']}")
            continue
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        plt.figure(figsize=(12, 8))
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.title(f"File: {os.path.basename(d['file_name'])}")
        plt.axis('off')
        plt.show()

def visualize_specific_class(dataset_dicts, metadata, target_class_id):
    """Find and visualize an image containing a specific class."""
    print(f"Searching for images containing Class ID: {target_class_id}...")
    for d in dataset_dicts:
        has_class = any(ann["category_id"] == target_class_id for ann in d["annotations"])
        if has_class:
            print(f"Found class {target_class_id} in image: {d['file_name']}")
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8)
            vis = visualizer.draw_dataset_dict(d)
            cv2_imshow(vis.get_image()[:, :, ::-1])
            return
    print(f"No images found with Class ID {target_class_id}")

visualize_random_samples(train_dataset_dicts, train_metadata, num_samples=2)
visualize_specific_class(train_dataset_dicts, train_metadata, target_class_id=3)

# ## Augmentations
def get_train_augmentation():
    """
    Define the Albumentations augmentation pipeline for training.
    These augmentations are specifically designed for histopathology/microscopy images.

    Returns:
        A.Compose: Albumentations composition with bbox support
    """
    transform = A.Compose([
        # ============ GEOMETRIC AUGMENTATIONS ============
        # Flips (very safe, preserves cell morphology)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # Rotation (90-degree rotations are safe for cells)
        A.RandomRotate90(p=0.5),

        # Slight rotation for more variety
        A.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_REFLECT),

        # Affine transformations (scale, translate, shear)
        A.Affine(
            scale=(0.9, 1.1),  # 90% to 110% zoom
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},  # Small shifts
            shear=(-5, 5),  # Small shear
            p=0.3,
            mode=cv2.BORDER_REFLECT
        ),

        # ============ COLOR/INTENSITY AUGMENTATIONS ============
        # Important for histopathology due to staining variations
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=1.0
            ),
        ], p=0.5),

        # Color jitter for staining variation robustness
        A.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05,
            p=0.3
        ),

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Good for microscopy images
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),

        # ============ BLUR/NOISE AUGMENTATIONS ============
        # Simulate focus variations and sensor noise
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.2),

        A.GaussNoise(std_range=(0.02, 0.05), p=0.2),

        # ============ ADVANCED AUGMENTATIONS ============
        # Elastic transform (slight tissue deformation simulation)
        A.ElasticTransform(
            alpha=50,
            sigma=5,
            p=0.1,
        ),

        # GridDistortion (simulate microscope distortions)
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.1),

    ], bbox_params=A.BboxParams(
        format='coco',  # COCO format: [x_min, y_min, width, height]
        label_fields=['category_ids'],
        min_area=1,  # Remove very small boxes after augmentation
        min_visibility=0.1,  # Keep boxes that are at least 10% visible
    ))

    return transform


def get_validation_augmentation():
    """
    Minimal augmentation for validation (usually just normalization).
    We don't use heavy augmentation during validation.
    """
    transform = A.Compose([
        # No augmentation for validation
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['category_ids'],
    ))

    return transform


class AlbumentationsMapper:
    """
    Custom DatasetMapper that applies Albumentations augmentations.

    This mapper:
    1. Loads images and annotations from the dataset
    2. Applies Albumentations transforms to both image and bboxes
    3. Converts the result back to Detectron2 format
    """

    def __init__(self, cfg, is_train=True):
        self.cfg = cfg
        self.is_train = is_train

        # Get augmentation pipeline
        if is_train:
            self.augmentation = get_train_augmentation()
        else:
            self.augmentation = get_validation_augmentation()

        # Image format (BGR for OpenCV/Detectron2)
        self.image_format = cfg.INPUT.FORMAT

    def __call__(self, dataset_dict):
        """
        Apply augmentations to a single data sample.

        Args:
            dataset_dict: A dict containing image and annotation info

        Returns:
            Augmented dataset_dict in Detectron2 format
        """
        from detectron2.structures import BoxMode

        # Make a deep copy to avoid modifying the original
        dataset_dict = copy.deepcopy(dataset_dict)

        # Load image
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)

        # Get annotations
        annotations = dataset_dict.get("annotations", [])

        if len(annotations) > 0:
            # Convert annotations to Albumentations format
            # COCO format: [x_min, y_min, width, height]
            bboxes = []
            category_ids = []
            masks = []

            for ann in annotations:
                bbox = ann["bbox"]
                # Ensure bbox is in COCO format [x, y, w, h]
                if ann.get("bbox_mode") == BoxMode.XYXY_ABS:
                    # Convert from [x1, y1, x2, y2] to [x, y, w, h]
                    bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

                bboxes.append(bbox)
                category_ids.append(ann["category_id"])

                # Convert polygon to binary mask
                # ann["segmentation"] is a list of polygons (list of lists)
                h, w = image.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                for poly in ann["segmentation"]:
                    pts = np.array(poly).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
                masks.append(mask)

            # Apply Albumentations
            try:
                # Convert BGR to RGB for Albumentations
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

                # Get transformed results
                image_aug = transformed["image"]
                bboxes_aug = transformed["bboxes"]
                category_ids_aug = transformed["category_ids"]
                masks_aug = transformed["masks"]

                # Convert back to BGR if needed
                if self.image_format == "BGR":
                    image_aug = image_aug[:, :, ::-1]

                image = image_aug

                # Rebuild annotations with transformed bboxes and masks
                new_annotations = []
                for bbox, cat_id, mask_aug in zip(bboxes_aug, category_ids_aug, masks_aug):
                    # Convert mask back to polygon
                    # Find contours in the binary mask
                    contours, _ = cv2.findContours(mask_aug.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    segmentation = []
                    for contour in contours:
                        if contour.size >= 6: # Need at least 3 points (6 coords) for a polygon
                            segmentation.append(contour.flatten().tolist())

                    # Skip if no valid polygon found (e.g. object was augmented out)
                    if not segmentation:
                        continue

                    ann = {
                        "bbox": list(bbox),
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": cat_id,
                        "segmentation": segmentation,
                        "iscrowd": 0,
                    }
                    new_annotations.append(ann)

                annotations = new_annotations

            except Exception as e:
                # If augmentation fails (e.g., all boxes removed), use original
                print(f"Warning: Augmentation failed for {dataset_dict['file_name']}: {e}")
                pass

        # Convert image to tensor format expected by Detectron2
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        # Convert annotations to Detectron2 Instances format
        if not self.is_train:
            # For validation, we might not need annotations
            dataset_dict.pop("annotations", None)
            return dataset_dict

        # Create instances from annotations
        annos = [
            utils.transform_instance_annotations(
                obj, [], image.shape[:2]
            )
            for obj in annotations
        ]

        instances = utils.annotations_to_instances(
            annos, image.shape[:2]
        )

        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict


class TrainerWithAlbumentations(DefaultTrainer):
    """
    Custom trainer that uses Albumentations for data augmentation.
    """

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Build a custom train loader with Albumentations augmentations.
        """
        mapper = AlbumentationsMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)


# ## Preprocessing
cfg = get_cfg()
cfg.OUTPUT_DIR = "/content/drive/MyDrive/ColabNotebooks/models/Detectron2_Models"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# DATASET
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_validation",)
cfg.DATALOADER.NUM_WORKERS = 2

# MODEL WEIGHTS & SOLVER
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 4  # Increased from 2 to stabilize gradients (use gradient accumulation if OOM)
cfg.SOLVER.BASE_LR = 0.001    # Increased LR slightly (default is often 0.00025 or 0.001)
cfg.SOLVER.MAX_ITER = 5000    # Increased from 1000 to allow convergence with augmentation
cfg.SOLVER.STEPS = [1000, 2000] # Decay LR at these steps
cfg.SOLVER.GAMMA = 0.5        # Decay factor

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # Back to default 512 to reduce False Positives
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5

# OPTIMIZATION FOR SMALL OBJECTS (TUMOR BUDS)
# 1. Anchor sizes: Default is [[32], [64], [128], [256], [512]]
#    Tumor buds are very small, so we shift anchors down.
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]

# 2. RPN (Region Proposal Network) settings
#    Lower thresholds to allow more candidate boxes for small objects
cfg.MODEL.RPN.NMS_THRESH = 0.7
cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]

# 3. ROI Heads (The classifier part)
#    Lower the threshold for what counts as a "positive" sample during training
#    Default is 0.5. Lowering to 0.4 helps it learn from slightly misaligned boxes.
cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.4]

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# ## Training
trainer = TrainerWithAlbumentations(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# ## Configs
config_yaml_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
with open(config_yaml_path, 'w') as file:
    yaml.dump(cfg, file)
print(f"Configuration saved to: {config_yaml_path}")

# ## Evalutation
def dice_coefficient(pred_count, gt_count):
    """
    Calculate DICE coefficient for counting.
    DICE = 2 * min(pred, gt) / (pred + gt)

    Returns 1.0 for perfect match, 0.0 for complete mismatch.
    """
    if pred_count + gt_count == 0:
        return 1.0  # Both are 0, perfect match
    return (2 * min(pred_count, gt_count)) / (pred_count + gt_count)


def count_based_dice(pred_counts, gt_counts, num_classes):
    """
    Calculate DICE coefficient based on detection counts per class.
    This measures how well the model's detection counts match ground truth counts.
    """
    dice_per_class = {}
    for class_id in range(num_classes):
        pred_n = pred_counts.get(class_id, 0)
        gt_n = gt_counts.get(class_id, 0)
        dice_per_class[class_id] = dice_coefficient(pred_n, gt_n)
    return dice_per_class


def match_detections_to_ground_truth(pred_boxes, pred_classes, gt_boxes, gt_classes, iou_threshold=0.3):
    """
    Match predicted detections to ground truth using IoU.
    This is used to determine True Positives, False Positives, and False Negatives.

    We use a low IoU threshold (0.3) because we only care if the bud was detected,
    not if the bounding box is perfectly aligned.

    Returns:
        matched_gt: set of ground truth indices that were matched
        matched_pred: set of prediction indices that were matched
        tp: number of true positives
        fp: number of false positives
        fn: number of false negatives
    """
    matched_gt = set()
    matched_pred = set()

    # Calculate IoU for all pairs
    for pred_idx, (pred_box, pred_cls) in enumerate(zip(pred_boxes, pred_classes)):
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
            # Only match same class
            if pred_cls != gt_cls:
                continue
            # Skip already matched ground truths
            if gt_idx in matched_gt:
                continue

            # Calculate IoU
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # If we found a match above threshold
        if best_iou >= iou_threshold and best_gt_idx != -1:
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)

    tp = len(matched_pred)  # True Positives: predictions that matched GT
    fp = len(pred_boxes) - tp  # False Positives: predictions without matching GT
    fn = len(gt_boxes) - len(matched_gt)  # False Negatives: GT without matching pred

    return matched_gt, matched_pred, tp, fp, fn


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two boxes.
    Boxes are in format [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union


def evaluate_detection_metrics(dataset_dicts, predictor, class_names, iou_threshold=0.3):
    """
    Evaluate detection performance using detection-focused metrics.

    This function evaluates:
    1. Per-class and overall Precision, Recall, F1
    2. Per-class and overall DICE coefficient (based on counts)
    3. Per-class and overall MAE (Mean Absolute Error for counting)
    4. Detection Rate (percentage of GT objects detected)

    Args:
        dataset_dicts: List of dataset dictionaries
        predictor: Detectron2 predictor
        class_names: List of class names
        iou_threshold: IoU threshold for matching (low value since we care about detection, not localization)

    Returns:
        DataFrame with per-image results
    """
    print(f"\n{'='*70}")
    print("TUMOUR BUDDING DETECTION EVALUATION")
    print(f"Evaluating on {len(dataset_dicts)} images...")
    print(f"IoU Threshold for matching: {iou_threshold}")
    print(f"{'='*70}\n")

    num_classes = len(class_names)

    # Aggregated metrics
    total_tp_per_class = {i: 0 for i in range(num_classes)}
    total_fp_per_class = {i: 0 for i in range(num_classes)}
    total_fn_per_class = {i: 0 for i in range(num_classes)}
    total_gt_per_class = {i: 0 for i in range(num_classes)}
    total_pred_per_class = {i: 0 for i in range(num_classes)}
    mae_errors_per_class = {i: [] for i in range(num_classes)}
    dice_per_class_list = {i: [] for i in range(num_classes)}

    results_table = []

    for d in dataset_dicts:
        # Get ground truth
        gt_annotations = d["annotations"]
        gt_boxes = [ann["bbox"] for ann in gt_annotations]
        # Convert from [x, y, w, h] to [x1, y1, x2, y2] if needed
        gt_boxes = [[b[0], b[1], b[0]+b[2], b[1]+b[3]] if len(b) == 4 and b[2] < 500 else b for b in gt_boxes]
        gt_classes = [ann["category_id"] for ann in gt_annotations]
        gt_counts = Counter(gt_classes)

        # Get predictions
        im = cv2.imread(d["file_name"])
        if im is None:
            print(f"Warning: Could not read {d['file_name']}")
            continue

        outputs = predictor(im)
        instances = outputs["instances"].to("cpu")

        pred_boxes = instances.pred_boxes.tensor.numpy().tolist() if len(instances) > 0 else []
        pred_classes = instances.pred_classes.numpy().tolist() if len(instances) > 0 else []
        pred_counts = Counter(pred_classes)

        # Match detections to ground truth
        _, _, tp, fp, fn = match_detections_to_ground_truth(
            pred_boxes, pred_classes, gt_boxes, gt_classes, iou_threshold
        )

        # Calculate per-class metrics for this image
        file_result = {"file": os.path.basename(d["file_name"])}

        for class_id in range(num_classes):
            gt_n = gt_counts.get(class_id, 0)
            pred_n = pred_counts.get(class_id, 0)

            # Count-based metrics
            mae = abs(pred_n - gt_n)
            dice = dice_coefficient(pred_n, gt_n)

            # Update aggregates
            total_gt_per_class[class_id] += gt_n
            total_pred_per_class[class_id] += pred_n
            mae_errors_per_class[class_id].append(mae)
            dice_per_class_list[class_id].append(dice)

            # Store in file result
            class_name = class_names[class_id]
            file_result[f"{class_name}_GT"] = gt_n
            file_result[f"{class_name}_Pred"] = pred_n
            file_result[f"{class_name}_DICE"] = f"{dice:.3f}"

        # Per-class TP/FP/FN counting
        for class_id in range(num_classes):
            gt_class_boxes = [b for b, c in zip(gt_boxes, gt_classes) if c == class_id]
            pred_class_boxes = [b for b, c in zip(pred_boxes, pred_classes) if c == class_id]
            pred_class_classes = [c for c in pred_classes if c == class_id]
            gt_class_classes = [c for c in gt_classes if c == class_id]

            if len(gt_class_boxes) > 0 or len(pred_class_boxes) > 0:
                _, _, class_tp, class_fp, class_fn = match_detections_to_ground_truth(
                    pred_class_boxes, [class_id]*len(pred_class_boxes),
                    gt_class_boxes, [class_id]*len(gt_class_boxes),
                    iou_threshold
                )
                total_tp_per_class[class_id] += class_tp
                total_fp_per_class[class_id] += class_fp
                total_fn_per_class[class_id] += class_fn

        file_result["Total_GT"] = len(gt_boxes)
        file_result["Total_Pred"] = len(pred_boxes)
        results_table.append(file_result)

    # ==========================================
    # PRINT SUMMARY REPORT
    # ==========================================
    print("\n" + "="*70)
    print("DETECTION METRICS SUMMARY")
    print("="*70)
    print("\n[PRECISION, RECALL, F1-SCORE]")
    print("Precision: Of all detections, how many were correct?")
    print("Recall: Of all ground truth objects, how many were detected?")
    print("F1-Score: Harmonic mean of Precision and Recall")
    print("-"*70)
    print(f"{'Class':<15} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    print("-"*70)

    total_tp = sum(total_tp_per_class.values())
    total_fp = sum(total_fp_per_class.values())
    total_fn = sum(total_fn_per_class.values())

    for class_id in range(num_classes):
        tp = total_tp_per_class[class_id]
        fp = total_fp_per_class[class_id]
        fn = total_fn_per_class[class_id]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{class_names[class_id]:<15} | {tp:<6} | {fp:<6} | {fn:<6} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f}")

    # Overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    print("-"*70)
    print(f"{'OVERALL':<15} | {total_tp:<6} | {total_fp:<6} | {total_fn:<6} | {overall_precision:<10.4f} | {overall_recall:<10.4f} | {overall_f1:<10.4f}")
    print("="*70)

    # DICE and MAE Report
    print("\n" + "="*70)
    print("COUNTING ACCURACY (DICE & MAE)")
    print("="*70)
    print("DICE: Measures count similarity (1.0 = perfect, 0.0 = no overlap)")
    print("MAE: Mean Absolute Error (average count difference per image)")
    print("-"*70)
    print(f"{'Class':<15} | {'Total GT':<10} | {'Total Pred':<10} | {'Avg DICE':<10} | {'MAE':<10}")
    print("-"*70)

    total_gt_all = sum(total_gt_per_class.values())
    total_pred_all = sum(total_pred_per_class.values())
    all_dice = []
    all_mae = []

    for class_id in range(num_classes):
        avg_dice = np.mean(dice_per_class_list[class_id]) if dice_per_class_list[class_id] else 0
        avg_mae = np.mean(mae_errors_per_class[class_id]) if mae_errors_per_class[class_id] else 0
        all_dice.append(avg_dice)
        all_mae.append(avg_mae)

        print(f"{class_names[class_id]:<15} | {total_gt_per_class[class_id]:<10} | {total_pred_per_class[class_id]:<10} | {avg_dice:<10.4f} | {avg_mae:<10.4f}")

    overall_dice = np.mean(all_dice)
    overall_mae = np.mean(all_mae)

    print("-"*70)
    print(f"{'OVERALL':<15} | {total_gt_all:<10} | {total_pred_all:<10} | {overall_dice:<10.4f} | {overall_mae:<10.4f}")
    print("="*70)

    # Detection Rate
    detection_rate = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    print(f"\n{'='*70}")
    print(f"DETECTION RATE: {detection_rate:.2%}")
    print(f"({total_tp} out of {total_tp + total_fn} ground truth buds were detected)")
    print(f"{'='*70}")

    # Summary Box
    print("\n" + "="*70)
    print("FINAL EVALUATION SUMMARY")
    print("="*70)
    print(f"  Overall Precision:     {overall_precision:.4f}")
    print(f"  Overall Recall:        {overall_recall:.4f}")
    print(f"  Overall F1-Score:      {overall_f1:.4f}")
    print(f"  Overall DICE:          {overall_dice:.4f}")
    print(f"  Overall MAE:           {overall_mae:.4f}")
    print(f"  Detection Rate:        {detection_rate:.2%}")
    print("="*70)

    return pd.DataFrame(results_table)

# ## Evaluate
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

# --- DETECTION THRESHOLDS ---
# Score threshold: Lower = more detections (may include false positives)
# 0.4-0.5 is a good balance for cell detection
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4

# NMS threshold: Controls overlap tolerance
# Lower = stricter, removes more overlapping boxes
# Higher = more lenient, keeps more overlapping boxes
# If under-counting clusters: RAISE to 0.5
# If double-counting: LOWER to 0.2
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3

predictor = DefaultPredictor(cfg)

# Get class names from metadata
class_names = train_metadata.thing_classes
print(f"\nClass names: {class_names}")

# Run evaluation
df_results = evaluate_detection_metrics(
    val_dataset_dicts,
    predictor,
    class_names,
    iou_threshold=0.3  # Low threshold since we care about detection, not localization
)

# Save results to CSV
results_csv_path = os.path.join(cfg.OUTPUT_DIR, "detection_evaluation_results.csv")
df_results.to_csv(results_csv_path, index=False)
print(f"\nDetailed results saved to: {results_csv_path}")

# ## Visualize Predictions
print("\n" + "="*70)
print("VISUALIZATION OF PREDICTIONS")
print("="*70)

def visualize_predictions_with_counts(dataset_dicts, predictor, metadata, num_samples=3):
    """Visualize predictions with count comparison."""
    for d in random.sample(dataset_dicts, min(num_samples, len(dataset_dicts))):
        im = cv2.imread(d["file_name"])
        if im is None:
            continue

        outputs = predictor(im)

        # Calculate counts
        pred_classes = outputs["instances"].pred_classes.to("cpu").numpy()
        pred_counts = Counter(pred_classes)
        gt_counts = Counter([ann["category_id"] for ann in d["annotations"]])

        # Create status string
        status_str = "COUNTS: "
        for i, name in enumerate(class_names):
            if gt_counts[i] > 0 or pred_counts.get(i, 0) > 0:
                status_str += f"[{name} GT:{gt_counts[i]} Pred:{pred_counts.get(i, 0)}]  "

        print(f"\nFile: {os.path.basename(d['file_name'])}")
        print(status_str)

        v = Visualizer(
            im[:, :, ::-1],
            metadata=metadata,
            scale=0.8,
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2_imshow(out.get_image()[:, :, ::-1])

# Visualize predictions
visualize_predictions_with_counts(val_dataset_dicts, predictor, train_metadata, num_samples=3)