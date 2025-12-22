import openslide
import os
import cv2
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# --- CONFIGURATION ---
ROOT_DIR = "/home/yashasnadigsyn/Downloads/"
OUTPUT_DIR = "1446_H-514_24_bareilly"
TXT_FILE_PATH = "good_good_slides.txt"
CHECKPOINT_FILE = "processing_checkpoint.json"

# Tile Sizes
BIG_TILE_SIZE = 5120
SMALL_TILE_SIZE = 1024

# --- NEW ROBUST THRESHOLDS ---
SATURATION_THRESHOLD = 20  
BRIGHTNESS_THRESHOLD = 215
TISSUE_PERCENT_THRESHOLD = 0.20 

# ---------------------

def load_checkpoint():
    """Load the list of already processed slides"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
            return set(data.get('processed_slides', []))
    return set()

def save_checkpoint(processed_slides):
    """Save the current progress"""
    data = {
        'processed_slides': list(processed_slides),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_processed': len(processed_slides)
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Checkpoint saved: {len(processed_slides)} slides processed")

def generate_tissue_mask(slide):
    """
    STAGE A: Global Masking
    Generates a low-res map of the tissue using Otsu's thresholding.
    """
    level = slide.level_count - 1
    img_thumb = slide.read_region((0, 0), level, slide.level_dimensions[level])
    img_thumb = img_thumb.convert("RGB")
    img_np = np.array(img_thumb)
    
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return mask, level

def is_tile_worthy(img_np, tissue_pct=TISSUE_PERCENT_THRESHOLD):
    """
    STAGE B: Local Filtering
    Checks if an image is valid tissue using HSV (Color) and Brightness.
    """
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    white_mask = v_channel > BRIGHTNESS_THRESHOLD
    gray_mask = (s_channel < SATURATION_THRESHOLD) & (v_channel < BRIGHTNESS_THRESHOLD)
    tissue_mask = ~(white_mask | gray_mask)

    tissue_count = np.count_nonzero(tissue_mask)
    total_pixels = img_np.shape[0] * img_np.shape[1]
    
    return (tissue_count / total_pixels) > tissue_pct

def process_pipeline():
    # Load checkpoint
    processed_slides = load_checkpoint()
    if processed_slides:
        print(f"📌 Resuming: {len(processed_slides)} slides already processed")
    else:
        print("🆕 Starting fresh processing")
    
    # Load the list of good slides
    if os.path.exists(TXT_FILE_PATH):
        with open(TXT_FILE_PATH, 'r') as f:
            good_slides_names = set([line.strip() for line in f.readlines() if line.strip()])
        print(f"Found {len(good_slides_names)} slides in list.")
    else:
        print("Warning: TXT file not found. Processing all slides?")
        good_slides_names = None

    # Collect all slides to process
    slides_to_process = []
    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            if not file.lower().endswith(('.tif', '.tiff', '.svs', '.ndpi')):
                continue

            slide_name_no_ext = os.path.splitext(file)[0]

            # Filter by text file if provided
            if good_slides_names and (file not in good_slides_names and slide_name_no_ext not in good_slides_names):
                continue
            
            # Skip if already processed
            if slide_name_no_ext in processed_slides:
                continue

            slides_to_process.append((root, file, slide_name_no_ext))
    
    if not slides_to_process:
        print("✅ All slides already processed!")
        return
    
    print(f"\n📊 Remaining slides to process: {len(slides_to_process)}")
    print(f"📊 Already completed: {len(processed_slides)}")
    
    # Process each slide
    for idx, (root, file, slide_name_no_ext) in enumerate(slides_to_process, 1):
        slide_path = os.path.join(root, file)
        print(f"\n{'='*60}")
        print(f"Processing [{idx}/{len(slides_to_process)}]: {slide_name_no_ext}")
        print(f"{'='*60}")
        
        slide_output_folder = os.path.join(OUTPUT_DIR, slide_name_no_ext)
        os.makedirs(slide_output_folder, exist_ok=True)
        
        try:
            slide = openslide.OpenSlide(slide_path)
            w, h = slide.level_dimensions[0]
            
            # Generate global mask
            global_mask, mask_level = generate_tissue_mask(slide)
            mask_scale = slide.level_downsamples[mask_level]

            y_steps = range(0, h, BIG_TILE_SIZE)
            x_steps = range(0, w, BIG_TILE_SIZE)
            
            with tqdm(total=len(y_steps)*len(x_steps), desc=f"Tiling {slide_name_no_ext}") as pbar:
                for y in y_steps:
                    for x in x_steps:
                        
                        if x + BIG_TILE_SIZE > w or y + BIG_TILE_SIZE > h:
                            pbar.update(1)
                            continue

                        # Check global mask
                        mask_x = int(x / mask_scale)
                        mask_y = int(y / mask_scale)
                        
                        if mask_y < global_mask.shape[0] and mask_x < global_mask.shape[1]:
                            if global_mask[mask_y, mask_x] == 0:
                                pbar.update(1)
                                continue

                        # Read and filter big tile
                        big_img = slide.read_region((x, y), 0, (BIG_TILE_SIZE, BIG_TILE_SIZE))
                        big_img = big_img.convert("RGB")
                        big_img_np = np.array(big_img)
                        
                        if not is_tile_worthy(big_img_np, tissue_pct=TISSUE_PERCENT_THRESHOLD):
                            pbar.update(1)
                            continue
                        
                        # Save big tile
                        big_tile_name = f"tile_{x}_{y}_{BIG_TILE_SIZE}"
                        big_tile_path = os.path.join(slide_output_folder, big_tile_name + ".png")
                        cv2.imwrite(big_tile_path, cv2.cvtColor(big_img_np, cv2.COLOR_RGB2BGR))
                        
                        # Create sub-folder
                        sub_tile_folder = os.path.join(slide_output_folder, big_tile_name)
                        os.makedirs(sub_tile_folder, exist_ok=True)
                        
                        # Sub-tiling
                        for r in range(0, BIG_TILE_SIZE, SMALL_TILE_SIZE):
                            for c in range(0, BIG_TILE_SIZE, SMALL_TILE_SIZE):
                                small_tile_np = big_img_np[r:r+SMALL_TILE_SIZE, c:c+SMALL_TILE_SIZE, :]
                                
                                if not is_tile_worthy(small_tile_np, tissue_pct=0.05):
                                    continue

                                small_tile_name = f"subtile_{c}_{r}_{SMALL_TILE_SIZE}.png"
                                small_save_path = os.path.join(sub_tile_folder, small_tile_name)
                                cv2.imwrite(small_save_path, cv2.cvtColor(small_tile_np, cv2.COLOR_RGB2BGR))
                        
                        pbar.update(1)
            
            # Mark this slide as processed
            processed_slides.add(slide_name_no_ext)
            save_checkpoint(processed_slides)
            print(f"✅ Completed: {slide_name_no_ext}")
            
        except Exception as e:
            print(f"❌ Error reading {slide_name_no_ext}: {e}")
            print("Skipping this slide...")
    
    print(f"\n{'='*60}")
    print(f"🎉 Processing complete!")
    print(f"Total slides processed: {len(processed_slides)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    process_pipeline()