import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from torchvision.models import AlexNet_Weights

from preproc import preproc_image   # your OpenCV pipeline


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
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        vec = alexnet(tensor)
    return vec.squeeze(0).cpu().numpy()


# ---------------- Main ----------------

if __name__ == "__main__":

    PATH = Path(".")
    rows = []

    for folder in PATH.iterdir():
        if not folder.is_dir():
            continue

        label = folder.name

        for file in folder.iterdir():
            if not file.is_file():
                continue

            try:
                # Load PIL
                pil_img = Image.open(file).convert("RGB")

                # PIL → OpenCV
                cv_img = pil_to_cv(pil_img)

                # Your preprocessing pipeline
                cv_img = preproc_image(cv_img)

                # OpenCV → PIL
                pil_img = cv_to_pil(cv_img)

                # Feature extraction
                vec = img2vec(pil_img)

                row = {"name": label}
                for i, v in enumerate(vec):
                    row[f"v{i+1}"] = float(v)

                rows.append(row)

            except Exception as e:
                print(f"Skipping {file}: {e}")

    # ---------------- Export CSV ----------------

    df = pd.DataFrame(rows)
    df.to_csv("values.csv", index=False)

    print("values.csv written successfully")