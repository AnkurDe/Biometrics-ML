"""This has the image preprocessing pipeline"""
import sys
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import cv2
import numpy as np


def grey_scale(img):
    """Greyscale of image"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Iteration 3
def distortion_correction(img):
    """Remove artifacts like lines present in notebooks without affecting handwriting"""
    # Create horizontal kernel to detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))

    # Use morphological opening to detect horizontal lines
    lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Create mask for the lines
    _, mask = cv2.threshold(lines, 0, 255, cv2.THRESH_BINARY)

    # Inpaint the lines using surrounding pixels
    repaired = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    return repaired


def normalise_contrast(img):
    """Normalize contrast"""
    # _, mask = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

    mask = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # better for text (white background → black text)
        15,  # block size (tune this: 11–25)
        8  # constant (tune: 2–10)
    )

    return mask


def noise_reduction(img):
    """Noise Removal"""
    return cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=13, searchWindowSize=21)


# Iteration 3
# Corrects the rotation artifacts
def deskew(img):
    """Corrects rotation of image"""
    return img


def compress_to_min_dim(img, target_min=1200):
    """Compress image to target size"""
    h, w = img.shape[:2]

    # Find current minimum dimension
    min_dim = min(h, w)

    # If already smaller, do nothing
    if min_dim <= target_min:
        return img

    # Compute scale factor
    scale = target_min / min_dim

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return resized


# Iteration 2
# This divides image into parts based on how the image was taken
def segment(img):
    """Segments image into multiple segments"""

    img = compress_to_min_dim(img, target_min=1200)

    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Step 2: Find connected components (letters/parts)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)

    heights = []

    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]

        # Filter noise (very small components)
        if area > 50:
            heights.append(h)

    # Step 3: Estimate handwriting scale
    if len(heights) == 0:
        return [img]  # fallback

    median_height = int(np.median(heights))

    # Step 4: Define dynamic segment size
    seg_h = int(median_height * 5)  # covers ~1–2 lines
    seg_w = int(median_height * 10)  # covers word groups

    h, w = img.shape

    segm = []

    # Step 5: Sliding window segmentation
    for y in range(0, h, seg_h):
        for x in range(0, w, seg_w):
            patch = img[y:y + seg_h, x:x + seg_w]

            # Skip mostly empty patches
            if np.mean(patch) > 10:
                segm.append((x, y, patch))

    return segm


def resize_for_screen(img, max_width=800, max_height=800):
    """Function to help with resizing image for display"""
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def preproc_image(img):
    """Image processing pipeline"""
    img = grey_scale(img)
    # img = distortion_correction(img)
    img = noise_reduction(img)
    img = normalise_contrast(img)
    img = deskew(img)

    segm = segment(img)
    return segm


# For visualization and debugging
if __name__ == '__main__':
    Tk().withdraw()

    path = askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )

    if not path:
        print("No image selected.")
        sys.exit()

    original = cv2.imread(path)

    if original is None:
        print("Failed to load image.")
        sys.exit()

    segments = preproc_image(original)
    print(len(segments))

    cv2.imshow("Original", resize_for_screen(original))

    # Display segments dynamically
    for idx, (x, y, seg) in enumerate(segments):
        window_name = f"Segment {idx}"
        cv2.imshow(window_name, resize_for_screen(seg))

        # Optional: visualize bounding box on original
        cv2.rectangle(original, (x, y), (x + seg.shape[1], y + seg.shape[0]), (0, 255, 0), 1)

    cv2.imshow("Segmented View", resize_for_screen(original))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
