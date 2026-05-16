"""
AlexNet Feature Extraction  (incremental CSV writes — low RAM)
==========================
Extracts feature vectors from:
  - classifier[4]  → penultimate-2 layer  (4096-d)
  - classifier[6]  → penultimate-1 layer  (1000-d, pre-softmax logits)
  - softmax output → final output layer   (1000-d, probabilities)

Each row is written to the CSV immediately after processing, so only
one image's data is ever held in RAM at a time.
Columns: name, feat_0 … feat_6095
"""

import os
import csv
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = Path(__file__).parent.parent / "Processed_Data"
OUTPUT_CSV = Path(__file__).parent.parent / "TabulatedData" / "values_v6.csv"
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_DIM        = 4096 + 1000 + 1000   # 6096 total
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Standard ImageNet preprocessing expected by AlexNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def load_model():
    """Load pretrained AlexNet and set to eval mode."""
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    model.eval()
    return model


def extract_features(model, image_tensor):
    """
    Run a single image through AlexNet and return a flat numpy array:
      feat_0    … feat_4095  → classifier[4] ReLU output  (4096-d)
      feat_4096 … feat_5095  → classifier[6] logits       (1000-d)
      feat_5096 … feat_6095  → softmax probabilities      (1000-d)
    """
    penultimate2_vec = None

    def hook_fn(module, input, output):
        nonlocal penultimate2_vec
        penultimate2_vec = output.detach()

    hook = model.classifier[4].register_forward_hook(hook_fn)

    with torch.no_grad():
        logits     = model(image_tensor.unsqueeze(0))   # (1, 1000)
        output_vec = torch.softmax(logits, dim=1)       # (1, 1000)

    hook.remove()

    v1 = penultimate2_vec.squeeze(0).numpy()   # (4096,)
    v2 = logits.squeeze(0).numpy()             # (1000,)
    v3 = output_vec.squeeze(0).numpy()         # (1000,)

    return np.concatenate([v1, v2, v3])        # (6096,)


def build_header():
    return ["name"] + [f"feat_{i}" for i in range(FEATURE_DIM)]


def process_dataset(data_path, model, csv_writer):
    """Walk every person folder; write one CSV row per image immediately."""
    total_images = 0
    total_errors = 0

    person_folders = sorted([
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    ])

    if not person_folders:
        raise ValueError(f"No subfolders found in: {data_path}")

    for person_name in person_folders:
        person_dir  = os.path.join(data_path, person_name)
        image_files = sorted([
            f for f in os.listdir(person_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ])

        if not image_files:
            print(f"  [SKIP] No images in: {person_name}")
            continue

        print(f"Processing '{person_name}' — {len(image_files)} image(s)")

        for img_file in image_files:
            img_path = os.path.join(person_dir, img_file)
            try:
                img    = Image.open(img_path).convert("RGB")
                tensor = preprocess(img)
                feats  = extract_features(model, tensor)

                # Write one row; no lists accumulate in RAM
                csv_writer.writerow([person_name] + feats.tolist())
                total_images += 1

            except Exception as e:
                print(f"  [ERROR] {img_path}: {e}")
                total_errors += 1

    return total_images, total_errors


def main():
    print("Loading AlexNet (pretrained on ImageNet)...")
    model = load_model()

    print(f"Scanning dataset at: {os.path.abspath(DATA_PATH)}\n")

    with open(OUTPUT_CSV, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(build_header())           # write header once

        total_images, total_errors = process_dataset(DATA_PATH, model, writer)
        # csv_file is flushed and closed here automatically

    print(f"\nDone.")
    print(f"  Rows written : {total_images}")
    print(f"  Errors       : {total_errors}")
    print(f"  Output       : {OUTPUT_CSV}")
    print(f"\nColumn breakdown:")
    print(f"  feat_0    … feat_4095  → classifier[4] output  (4096-d, penultimate-2)")
    print(f"  feat_4096 … feat_5095  → classifier[6] logits  (1000-d, penultimate-1)")
    print(f"  feat_5096 … feat_6095  → softmax probabilities (1000-d, final output)")


if __name__ == "__main__":
    main()
