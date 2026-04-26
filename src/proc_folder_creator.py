"""
Batch preprocessing script:
- Traverses ../Data directory
- Applies preprocessing to each image
- Saves segmented outputs in mirrored folder structure
"""

import cv2
from pathlib import Path
from tqdm import tqdm

# Import your preprocessing pipeline
from preproc_pipeline import preproc_image


# INPUT / OUTPUT PATHS
input_root = Path(__file__).parent.parent / "Data"
output_root = Path(__file__).parent.parent / "Processed_Data"


def is_image_file(file_path):
    return file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]


def process_all_images():
    # Collect all image paths
    image_paths = [p for p in input_root.rglob("*") if p.is_file() and is_image_file(p)]

    print(f"Total images found: {len(image_paths)}")

    for img_path in tqdm(image_paths, desc="Processing Images"):
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Run preprocessing
            segments = preproc_image(img)

            # Create corresponding output directory
            relative_path = img_path.relative_to(input_root).parent
            save_dir = output_root / relative_path
            save_dir.mkdir(parents=True, exist_ok=True)

            # Base filename without extension
            base_name = img_path.stem

            # Save each segment
            for idx, (_, _, seg) in enumerate(segments):
                save_path = save_dir / f"{base_name}_seg{idx}.png"
                cv2.imwrite(str(save_path), seg)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    process_all_images()
