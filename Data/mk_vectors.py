import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import pandas as pd


alexnet = models.alexnet(pretrained=True)

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


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def img2vec(img):
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        vec = alexnet(tensor)
    return vec.squeeze(0).cpu().numpy()  # shape: (10,)


if __name__ == '__main__':
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
                img = Image.open(file).convert("RGB")
                vec = img2vec(img)

                row = {"name": label}
                for i, v in enumerate(vec):
                    row[f"v{i+1}"] = float(v)

                rows.append(row)

            except Exception as e:
                print(f"Skipping {file}: {e}")

    # ------------------ Export CSV ------------------

    df = pd.DataFrame(rows)
    df.to_csv("values.csv", index=False)

    print("values.csv written successfully")
