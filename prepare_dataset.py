"""
Dataset Preparation Script for Tumor Budding Detection

This script:
1. Collects all subtile images from multiple tile folders
2. Renames files with parent tile prefix to avoid filename collisions
3. Merges all COCO annotations into unified JSON files
4. Randomly splits data into train (80%) and test (20%) sets
"""

import os
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple

# Set random seed for reproducibility
random.seed(42)

# Configuration
SOURCE_DIR = Path("AiML Tumor Budding Annotated")
OUTPUT_DIR = Path("dataset")
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2


def find_tile_folders(source_dir: Path) -> List[Path]:
    """Find all tile folders containing annotations."""
    tile_folders = []
    for item in source_dir.iterdir():
        if item.is_dir() and item.name.startswith("tile_"):
            # Check if it has a JSON file
            json_files = list(item.glob("*.json"))
            if json_files:
                tile_folders.append(item)
    return sorted(tile_folders)


def load_coco_annotations(json_path: Path) -> Dict:
    """Load COCO annotations from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def collect_all_data(tile_folders: List[Path]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Collect all images and annotations from all tile folders.
    Returns: (all_images, all_annotations, categories)
    """
    all_images = []
    all_annotations = []
    categories = None
    
    # We need to reassign IDs globally
    global_image_id = 1
    global_annotation_id = 0
    
    for tile_folder in tile_folders:
        tile_name = tile_folder.name
        print(f"Processing: {tile_name}")
        
        # Find the JSON file in this folder
        json_files = list(tile_folder.glob("*.json"))
        if not json_files:
            print(f"  Warning: No JSON file found in {tile_name}")
            continue
        
        json_path = json_files[0]
        coco_data = load_coco_annotations(json_path)
        
        # Get categories (should be same across all files)
        if categories is None:
            categories = coco_data.get("categories", [])
        
        # Create mapping from old image_id to new image_id
        old_to_new_image_id = {}
        
        # Process images
        for img in coco_data.get("images", []):
            old_image_id = img["id"]
            old_filename = img["file_name"]
            
            # Create new unique filename with tile prefix
            new_filename = f"{tile_name}_{old_filename}"
            
            # Create new image entry with global ID
            new_image = {
                "id": global_image_id,
                "width": img["width"],
                "height": img["height"],
                "file_name": new_filename,
                "original_folder": tile_name,
                "original_filename": old_filename
            }
            
            all_images.append(new_image)
            old_to_new_image_id[old_image_id] = global_image_id
            global_image_id += 1
        
        # Process annotations
        for ann in coco_data.get("annotations", []):
            old_image_id = ann["image_id"]
            
            # Skip if image was not found
            if old_image_id not in old_to_new_image_id:
                continue
            
            new_annotation = {
                "id": global_annotation_id,
                "image_id": old_to_new_image_id[old_image_id],
                "category_id": ann["category_id"],
                "segmentation": ann.get("segmentation", []),
                "bbox": ann.get("bbox", []),
                "area": ann.get("area", 0),
                "iscrowd": ann.get("iscrowd", 0)
            }
            
            all_annotations.append(new_annotation)
            global_annotation_id += 1
    
    print(f"\nTotal images collected: {len(all_images)}")
    print(f"Total annotations collected: {len(all_annotations)}")
    
    return all_images, all_annotations, categories


def split_dataset(
    all_images: List[Dict],
    all_annotations: List[Dict],
    train_ratio: float = 0.8
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Randomly split images and their annotations into train/test sets.
    """
    # Shuffle images randomly
    shuffled_images = all_images.copy()
    random.shuffle(shuffled_images)
    
    # Split point
    split_idx = int(len(shuffled_images) * train_ratio)
    
    train_images = shuffled_images[:split_idx]
    test_images = shuffled_images[split_idx:]
    
    # Create sets of image IDs for quick lookup
    train_image_ids = {img["id"] for img in train_images}
    test_image_ids = {img["id"] for img in test_images}
    
    # Split annotations based on image IDs
    train_annotations = [ann for ann in all_annotations if ann["image_id"] in train_image_ids]
    test_annotations = [ann for ann in all_annotations if ann["image_id"] in test_image_ids]
    
    # Reassign IDs for clean output
    # Reassign image IDs
    train_id_mapping = {}
    for new_id, img in enumerate(train_images, start=1):
        train_id_mapping[img["id"]] = new_id
        img["id"] = new_id
    
    test_id_mapping = {}
    for new_id, img in enumerate(test_images, start=1):
        test_id_mapping[img["id"]] = new_id
        img["id"] = new_id
    
    # Update annotation image_ids
    for ann in train_annotations:
        ann["image_id"] = train_id_mapping[ann["image_id"]]
    
    for ann in test_annotations:
        ann["image_id"] = test_id_mapping[ann["image_id"]]
    
    # Reassign annotation IDs
    for new_id, ann in enumerate(train_annotations):
        ann["id"] = new_id
    
    for new_id, ann in enumerate(test_annotations):
        ann["id"] = new_id
    
    print(f"\nTrain set: {len(train_images)} images, {len(train_annotations)} annotations")
    print(f"Test set: {len(test_images)} images, {len(test_annotations)} annotations")
    
    return train_images, train_annotations, test_images, test_annotations


def create_coco_json(images: List[Dict], annotations: List[Dict], categories: List[Dict], description: str) -> Dict:
    """Create a COCO format JSON structure."""
    # Remove helper fields from images
    clean_images = []
    for img in images:
        clean_img = {
            "id": img["id"],
            "width": img["width"],
            "height": img["height"],
            "file_name": img["file_name"]
        }
        clean_images.append(clean_img)
    
    return {
        "info": {
            "description": description,
            "version": "1.0",
            "year": 2025
        },
        "images": clean_images,
        "annotations": annotations,
        "categories": categories
    }


def copy_images(images: List[Dict], source_dir: Path, dest_dir: Path) -> None:
    """Copy images to destination directory with new names."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    for img in images:
        original_folder = img["original_folder"]
        original_filename = img["original_filename"]
        new_filename = img["file_name"]
        
        src_path = source_dir / original_folder / original_filename
        dst_path = dest_dir / new_filename
        
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Warning: Source file not found: {src_path}")


def main():
    print("=" * 60)
    print("Dataset Preparation for Tumor Budding Detection")
    print("=" * 60)
    
    # Create output directories
    train_dir = OUTPUT_DIR / "train"
    test_dir = OUTPUT_DIR / "test"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all tile folders
    print(f"\nSource directory: {SOURCE_DIR}")
    tile_folders = find_tile_folders(SOURCE_DIR)
    print(f"Found {len(tile_folders)} tile folders")
    
    # Collect all data
    print("\n" + "-" * 40)
    print("Collecting data from all folders...")
    print("-" * 40)
    all_images, all_annotations, categories = collect_all_data(tile_folders)
    
    if not all_images:
        print("Error: No images found!")
        return
    
    # Split dataset
    print("\n" + "-" * 40)
    print(f"Splitting dataset ({TRAIN_RATIO*100:.0f}% train, {TEST_RATIO*100:.0f}% test)...")
    print("-" * 40)
    train_images, train_annotations, test_images, test_annotations = split_dataset(
        all_images, all_annotations, TRAIN_RATIO
    )
    
    # Create COCO JSON files
    print("\n" + "-" * 40)
    print("Creating COCO annotation files...")
    print("-" * 40)
    
    train_coco = create_coco_json(train_images, train_annotations, categories, "Tumor Budding - Train Set")
    test_coco = create_coco_json(test_images, test_annotations, categories, "Tumor Budding - Test Set")
    
    # Save JSON files
    train_json_path = train_dir / "annotations.json"
    test_json_path = test_dir / "annotations.json"
    
    with open(train_json_path, 'w') as f:
        json.dump(train_coco, f, indent=2)
    print(f"Saved: {train_json_path}")
    
    with open(test_json_path, 'w') as f:
        json.dump(test_coco, f, indent=2)
    print(f"Saved: {test_json_path}")
    
    # Copy images
    print("\n" + "-" * 40)
    print("Copying images...")
    print("-" * 40)
    
    print("Copying train images...")
    copy_images(train_images, SOURCE_DIR, train_dir)
    
    print("Copying test images...")
    copy_images(test_images, SOURCE_DIR, test_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"\nTrain folder: {train_dir}")
    print(f"  - Images: {len(train_images)}")
    print(f"  - Annotations: {len(train_annotations)}")
    print(f"  - JSON: annotations.json")
    print(f"\nTest folder: {test_dir}")
    print(f"  - Images: {len(test_images)}")
    print(f"  - Annotations: {len(test_annotations)}")
    print(f"  - JSON: annotations.json")
    print(f"\nCategories: {[c['name'] for c in categories]}")


if __name__ == "__main__":
    main()
