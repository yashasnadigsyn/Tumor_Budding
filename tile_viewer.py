#!/usr/bin/env python3
"""
tile_viewer.py
Standalone script:
 - shows a GUI file picker to choose an image
 - draws SMALL_TILE_SIZE grid (default 1024) on the image
 - labels each subtile like subtile_{col}_{row}_{size}.png and coordinates
 - opens an interactive OpenCV window where you can:
     * drag with left mouse button to pan
     * press '+' or '=' to zoom in, '-' to zoom out
     * press 's' to save annotated preview PNG next to original
     * press 'q' or ESC to quit
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import math
import os
import sys
import tkinter as tk
from tkinter import filedialog

# CONFIG
SMALL_TILE_SIZE = 1024
OUTLINE_WIDTH = 2
TEXT_MARGIN = 6
WINDOW_NAME = "Tile Viewer (drag to pan, +/- to zoom, s=save, q=quit)"
DEFAULT_SAVE_SUFFIX = "_overlay.png"

def create_grid_overlay_pil(img_np, small_tile_size=1024, outline_width=2,
                            text_margin=6, font_path=None):
    """Return a PIL Image (RGBA) with grid overlay and labels."""
    img = Image.fromarray(img_np.astype('uint8'))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default() if font_path is None else ImageFont.truetype(font_path, 14)
    except Exception:
        font = ImageFont.load_default()

    H, W = img_np.shape[:2]
    n_cols = math.ceil(W / small_tile_size)
    n_rows = math.ceil(H / small_tile_size)

    # Grid lines
    for cx in range(n_cols + 1):
        x = min(cx * small_tile_size, W)
        draw.line([(x, 0), (x, H)], width=outline_width)

    for ry in range(n_rows + 1):
        y = min(ry * small_tile_size, H)
        draw.line([(0, y), (W, y)], width=outline_width)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    for r in range(n_rows):
        for c in range(n_cols):
            x0 = c * small_tile_size
            y0 = r * small_tile_size
            x1 = min((c + 1) * small_tile_size, W)
            y1 = min((r + 1) * small_tile_size, H)

            label = f"subtile_{c}_{r}_{small_tile_size}.png"
            coord = f"{x0},{y0}"
            text = f"{label}\n{coord}"

            bbox = odraw.multiline_textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            box_x1 = min(x0 + text_w + 2 * text_margin, x1 - 4)
            box_y1 = min(y0 + text_h + 2 * text_margin, y1 - 4)

            odraw.rectangle(
                [(x0 + 2, y0 + 2), (box_x1, box_y1)],
                fill=(0, 0, 0, 150)
            )

            odraw.multiline_text(
                (x0 + text_margin + 2, y0 + text_margin + 2),
                text,
                fill=(255, 255, 255, 255),
                font=font
            )

    return Image.alpha_composite(img.convert("RGBA"), overlay)

# ----------------- Interactive viewer helpers -----------------
class PanZoomViewer:
    def __init__(self, img_rgb):
        # Original high-res RGB numpy image
        self.orig = img_rgb
        self.H, self.W = self.orig.shape[:2]
        # view parameters
        self.scale = 1.0
        self.min_scale = max(0.1, min(800 / self.W, 600 / self.H))
        self.max_scale = 8.0
        self.offset = np.array([0.0, 0.0])  # top-left offset in original image coords
        self.dragging = False
        self.last_mouse = (0, 0)
        # window size suggestion
        self.win_w = min(self.W, 1400)
        self.win_h = min(self.H, 900)

    def start(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_NAME, self.win_w, self.win_h)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_cb)
        self.loop()

    def _mouse_cb(self, event, x, y, flags, param):
        # convert window coords (x,y) to image coords considering current scale & offset
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_mouse = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.last_mouse[0]
            dy = y - self.last_mouse[1]
            # move offset opposite to mouse movement (dragging canvas)
            self.offset -= np.array([dx, dy]) / self.scale
            # clamp offsets so we don't go too far
            self._clamp_offset()
            self.last_mouse = (x, y)
        # Note: mouse wheel events aren't consistent cross-platform, use keyboard +/- for zoom

    def _clamp_offset(self):
        max_x = max(0, self.W - (self.win_w / self.scale))
        max_y = max(0, self.H - (self.win_h / self.scale))
        self.offset[0] = float(np.clip(self.offset[0], 0, max_x))
        self.offset[1] = float(np.clip(self.offset[1], 0, max_y))

    def render(self):
        # compute viewport in original image coords
        vw = int(self.win_w / self.scale)
        vh = int(self.win_h / self.scale)
        x0 = int(self.offset[0])
        y0 = int(self.offset[1])
        x1 = min(x0 + vw, self.W)
        y1 = min(y0 + vh, self.H)

        # crop and resize to window
        crop = self.orig[y0:y1, x0:x1]
        if crop.size == 0:
            return np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)
        disp = cv2.resize(crop, (self.win_w, self.win_h), interpolation=cv2.INTER_LINEAR)
        # add overlay text for scale & coords
        info = f"scale={self.scale:.2f} view={x0},{y0}"
        cv2.putText(disp, info, (10, self.win_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2, cv2.LINE_AA)
        return disp

    def zoom_at_center(self, factor):
        old_scale = self.scale
        self.scale = float(np.clip(self.scale * factor, self.min_scale, self.max_scale))
        # adjust offset so zoom focuses on the center of the window
        center = np.array([self.win_w/2, self.win_h/2])
        center_in_orig = self.offset + center / old_scale
        self.offset = center_in_orig - center / self.scale
        self._clamp_offset()

    def loop(self):
        while True:
            disp = self.render()
            cv2.imshow(WINDOW_NAME, disp)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('+') or key == ord('='):
                self.zoom_at_center(1.25)
            elif key == ord('-') or key == ord('_'):
                self.zoom_at_center(1/1.25)
            elif key == ord('s'):
                # save current annotated full-res image
                # handled externally
                return 'save'
            # Continue looping otherwise
        cv2.destroyAllWindows()
        return 'quit'

# ----------------- Main -----------------
def pick_file():
    root = tk.Tk()
    root.withdraw()
    root.update()
    filetypes = [
        ("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"),
        ("All files", "*.*")
    ]
    path = filedialog.askopenfilename(title="Choose an image or tile PNG", filetypes=filetypes)
    root.destroy()
    return path

def load_image_any(path):
    # If openslide is installed and file extension is whole-slide, you could read a region here.
    # For now just use PIL to open regular raster images.
    try:
        pil = Image.open(path)
        pil = pil.convert("RGB")
        arr = np.array(pil)
        return arr
    except Exception as e:
        print("Error loading with PIL:", e)
        raise

def main():
    print("Select an image file (e.g. a 5120x5120 big tile PNG)")
    path = pick_file()
    if not path:
        print("No file selected. Exiting.")
        return

    print("Loading:", path)
    img_np = load_image_any(path)
    H, W = img_np.shape[:2]
    print(f"Image size: {W}x{H}")

    # Create annotated overlay (PIL)
    pil_overlay = create_grid_overlay_pil(img_np, small_tile_size=SMALL_TILE_SIZE)
    # Convert to RGB numpy for viewer
    overlay_rgb = np.array(pil_overlay.convert("RGB"))

    viewer = PanZoomViewer(overlay_rgb)
    action = viewer.start()  # will open window and block until quit
    # Note: the viewer's loop returns on 'save' but in current implementation it returns via loop result
    # To allow saving from viewer, we instead run loop manually to capture 's' key.
    # So modify loop usage:
    # We'll re-run a small loop to capture 's' if needed:
    # (Better approach: show again and capture 's' result)
    # For simplicity: reopen viewer but check for 's' inside loop:
    # Reopen interactive loop to allow save feature.
    # (Implement quick loop here:)

    # Start interactive loop with save support
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, viewer.win_w, viewer.win_h)
    cv2.setMouseCallback(WINDOW_NAME, viewer._mouse_cb)
    while True:
        disp = viewer.render()
        cv2.imshow(WINDOW_NAME, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('+') or key == ord('='):
            viewer.zoom_at_center(1.25)
        elif key == ord('-') or key == ord('_'):
            viewer.zoom_at_center(1/1.25)
        elif key == ord('s'):
            # save full-res overlay next to original file
            dirn, base = os.path.split(path)
            name, ext = os.path.splitext(base)
            save_path = os.path.join(dirn, name + DEFAULT_SAVE_SUFFIX)
            pil_overlay.convert("RGB").save(save_path, format="PNG")
            print("Saved overlay to:", save_path)
        # else continue loop
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
