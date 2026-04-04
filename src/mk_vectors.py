from pathlib import Path

import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import cv2
from torchvision.models import AlexNet_Weights
from tqdm import tqdm

from preproc import preproc_image


# ---------------- Device ----------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ---------------- Model ----------------

alexnet = models.alexnet(weights=AlexNet_Weights.DEFAULT)

alexnet.classifier = nn.Sequential(
    nn.Dropout(),
    nn.Linear(256 * 6 * 6, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 512),
    nn.ReLU(inplace=True),
    nn.Linear(512, 10)
)

alexnet = alexnet.to(device)
alexnet.eval()


# ---------------- Torch preprocessing ----------------

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ---------------- Helpers ----------------

def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv_to_pil(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def img2vec(img):
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = alexnet(tensor)
    return vec.squeeze(0).cpu().numpy()


def gpu_stats():
    if device.type != "cuda":
        return ""

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2

    return f"VRAM {allocated:.0f}/{reserved:.0f} MB"


# ---------------- Main ----------------

if __name__ == "__main__":

    PATH = Path("../Data")
    rows = []

    # Collect all files first (for accurate progress bar)
    all_files = [
        f for d in PATH.iterdir() if d.is_dir()
        for f in d.iterdir() if f.is_file()
    ]

    progress = tqdm(all_files, desc="Extracting", unit="img")

    for file in progress:

        label = file.parent.name

        try:
            # pil_img = cv2.imread(str(file))
            pil_img = Image.open(file).convert("RGB")

            cv_img = pil_to_cv(pil_img)
            cv_img = preproc_image(cv_img)
            pil_img = cv_to_pil(cv_img)

            vec = img2vec(pil_img)

            row = {"name": label}
            for i, v in enumerate(vec):
                row[f"v{i+1}"] = float(v)

            rows.append(row)

            if device.type == "cuda":
                progress.set_postfix_str(gpu_stats())

        except Exception as e:
            print(f"Skipping {file}: {e}")

    # ---------------- Export CSV ----------------

    df = pd.DataFrame(rows)
    df.to_csv("values.csv", index=False)

    print("values.csv written successfully")
