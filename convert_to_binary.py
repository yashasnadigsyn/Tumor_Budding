# ============================================================================
# CONVERT 5-CLASS ANNOTATIONS TO BINARY (TUMOR BUD VS BACKGROUND)
# ============================================================================
# This script converts the 5-class tumor bud annotations (1-cell, 2-cell, etc.)
# to a single "tumor_bud" class for binary classification.
# 
# Rationale: Instead of trying to classify the number of cells (which is 
# subjective even for pathologists), we focus on detecting whether something
# is a tumor bud or not. This gives us ~2000+ training samples instead of
# fighting severe class imbalance.
# ============================================================================

# %% [markdown]
# ## Cell 1: Imports and Configuration

# %%
import json
import os
import shutil
from pathlib import Path
from collections import Counter

# Configuration
BASE_DIR = Path(".")  # Change to your project root if needed
TRAIN_JSON = BASE_DIR / "dataset" / "train" / "annotations.json"
TEST_JSON = BASE_DIR / "dataset" / "test" / "annotations.json"
TRAIN_IMAGES = BASE_DIR / "dataset" / "train"
TEST_IMAGES = BASE_DIR / "dataset" / "test"

# Output directories
BINARY_DIR = BASE_DIR / "dataset_binary"
BINARY_TRAIN = BINARY_DIR / "train"
BINARY_TEST = BINARY_DIR / "test"

print(f"Source train JSON: {TRAIN_JSON}")
print(f"Source test JSON: {TEST_JSON}")
print(f"Output binary dataset: {BINARY_DIR}")

# %% [markdown]
# ## Cell 2: Helper Functions

# %%
def convert_to_binary_annotations(input_json_path: Path, output_json_path: Path):
    """
    Convert multi-class COCO annotations to binary classification.
    All category_ids (1-5) are mapped to a single class: 1 (tumor_bud)
    
    Args:
        input_json_path: Path to original annotations.json
        output_json_path: Path to save binary annotations.json
    
    Returns:
        dict with statistics about the conversion
    """
    # Load original annotations
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Store original class distribution
    original_counts = Counter(ann['category_id'] for ann in data['annotations'])
    print(f"\nOriginal class distribution:")
    for cat_id, count in sorted(original_counts.items()):
        print(f"  Class {cat_id}: {count} annotations")
    
    # Create new categories (single class)
    new_categories = [
        {"id": 1, "name": "tumor_bud", "supercategory": "tumor"}
    ]
    
    # Convert all annotations to class 1
    new_annotations = []
    for ann in data['annotations']:
        new_ann = ann.copy()
        new_ann['category_id'] = 1  # All tumor buds become class 1
        new_annotations.append(new_ann)
    
    # Create new data structure
    new_data = {
        "info": {
            "description": data.get('info', {}).get('description', 'Tumor Budding') + " - Binary Classification",
            "version": "2.0",
            "year": 2025
        },
        "images": data['images'],
        "annotations": new_annotations,
        "categories": new_categories
    }
    
    # Ensure output directory exists
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save binary annotations
    with open(output_json_path, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    print(f"\nBinary annotations saved to: {output_json_path}")
    print(f"Total annotations: {len(new_annotations)}")
    print(f"Single class: tumor_bud (id=1)")
    
    return {
        "original_counts": dict(original_counts),
        "total_annotations": len(new_annotations),
        "output_path": str(output_json_path)
    }

def copy_images(source_dir: Path, dest_dir: Path):
    """
    Copy all images from source to destination directory.
    Only copies image files (png, jpg, jpeg).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    copied_count = 0
    
    for src_file in source_dir.iterdir():
        if src_file.suffix.lower() in image_extensions:
            dest_file = dest_dir / src_file.name
            if not dest_file.exists():
                shutil.copy2(src_file, dest_file)
                copied_count += 1
    
    print(f"Copied {copied_count} images to {dest_dir}")
    return copied_count

# %% [markdown]
# ## Cell 3: Convert Training Set

# %%
print("=" * 60)
print("CONVERTING TRAINING SET TO BINARY")
print("=" * 60)

train_stats = convert_to_binary_annotations(
    TRAIN_JSON, 
    BINARY_TRAIN / "annotations.json"
)

# Copy training images
print("\nCopying training images...")
copy_images(TRAIN_IMAGES, BINARY_TRAIN)

# %% [markdown]
# ## Cell 4: Convert Test Set

# %%
print("\n" + "=" * 60)
print("CONVERTING TEST SET TO BINARY")
print("=" * 60)

test_stats = convert_to_binary_annotations(
    TEST_JSON,
    BINARY_TEST / "annotations.json"
)

# Copy test images
print("\nCopying test images...")
copy_images(TEST_IMAGES, BINARY_TEST)

# %% [markdown]
# ## Cell 5: Verify Conversion

# %%
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

# Verify train
with open(BINARY_TRAIN / "annotations.json", 'r') as f:
    binary_train = json.load(f)
print(f"\nBinary Train Dataset:")
print(f"  Images: {len(binary_train['images'])}")
print(f"  Annotations: {len(binary_train['annotations'])}")
print(f"  Categories: {binary_train['categories']}")
train_class_dist = Counter(ann['category_id'] for ann in binary_train['annotations'])
print(f"  Class distribution: {dict(train_class_dist)}")

# Verify test
with open(BINARY_TEST / "annotations.json", 'r') as f:
    binary_test = json.load(f)
print(f"\nBinary Test Dataset:")
print(f"  Images: {len(binary_test['images'])}")
print(f"  Annotations: {len(binary_test['annotations'])}")
print(f"  Categories: {binary_test['categories']}")
test_class_dist = Counter(ann['category_id'] for ann in binary_test['annotations'])
print(f"  Class distribution: {dict(test_class_dist)}")

# Count images in directories
train_images = list(BINARY_TRAIN.glob("*.png")) + list(BINARY_TRAIN.glob("*.jpg"))
test_images = list(BINARY_TEST.glob("*.png")) + list(BINARY_TEST.glob("*.jpg"))
print(f"\nImage files:")
print(f"  Train images: {len(train_images)}")
print(f"  Test images: {len(test_images)}")

print("\n" + "=" * 60)
print("CONVERSION COMPLETE!")
print("=" * 60)
print(f"\nBinary dataset created at: {BINARY_DIR.absolute()}")
print("\nYou can now use these paths for training:")
print(f"  TRAIN_JSON = '{BINARY_TRAIN / 'annotations.json'}'")
print(f"  TRAIN_IMAGES = '{BINARY_TRAIN}'")
print(f"  VAL_JSON = '{BINARY_TEST / 'annotations.json'}'")
print(f"  VAL_IMAGES = '{BINARY_TEST}'")
