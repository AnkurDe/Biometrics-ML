from pathlib import Path

import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import pandas as pd
from torchvision.models import AlexNet_Weights

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

# ---------------- Hook Storage ----------------

layer_outputs = {}


def get_hook(name):
    def hook(module, input, output):
        layer_outputs[name] = output.detach()

    return hook


# Register hooks for ALL layers
for name, layer in alexnet.features._modules.items():
    layer.register_forward_hook(get_hook(f"features_{name}"))

alexnet.avgpool.register_forward_hook(get_hook("avgpool"))

for name, layer in alexnet.classifier._modules.items():
    layer.register_forward_hook(get_hook(f"classifier_{name}"))


# ---------------- Feature Extraction ----------------

def img2vec_all(img):
    global layer_outputs
    layer_outputs = {}

    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        _ = alexnet(tensor)

    result = {}

    for layer_name, output in layer_outputs.items():

        # If Conv layer output → apply global average pooling
        if len(output.shape) == 4:
            flat = output.mean(dim=[2, 3]).view(-1).cpu().numpy()
        else:
            flat = output.view(-1).cpu().numpy()

        for i, v in enumerate(flat):
            result[f"{layer_name}_v{i + 1}"] = float(v)

    return result


# ---------------- GPU Stats ----------------

def gpu_stats():
    if device.type != "cuda":
        return ""

    allocated = torch.cuda.memory_allocated() / 1024 ** 2
    reserved = torch.cuda.memory_reserved() / 1024 ** 2

    return f"VRAM {allocated:.0f}/{reserved:.0f} MB"



# ---------------- CONFIG ----------------
BATCH_SIZE = 1000
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "values_all_layers.csv"
DATA_PATH = Path("../Processed_Data")


# ---------------- MAIN ----------------

def main():
    print(OUTPUT_FILE)
    input("")

    batch = []
    header_written = False

    # Collect all image files
    all_files = [
        f for d in DATA_PATH.iterdir() if d.is_dir()
        for f in d.iterdir() if f.is_file()
    ]

    progress = tqdm(all_files, desc="Extracting", unit="img")

    for file in progress:

        label = file.parent.name

        try:
            # Safe image handling
            with Image.open(file) as img:
                img = img.convert("RGB")

                # Feature extraction (your function)
                features = img2vec_all(img)

            # Prepare row
            row = {"name": label}
            row.update(features)

            batch.append(row)

            # Write batch when full
            if len(batch) >= BATCH_SIZE:
                write_batch(batch, header_written)
                header_written = True
                batch.clear()

            # Optional GPU stats
            if device.type == "cuda":
                progress.set_postfix_str(gpu_stats())
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Skipping {file}: {e}")

    # Write remaining data
    if batch:
        write_batch(batch, header_written)

    print(f"{OUTPUT_FILE} written successfully")


# ---------------- HELPER ----------------

def write_batch(batch, header_written):
    df_batch = pd.DataFrame(batch)

    if not header_written:
        df_batch.to_csv(OUTPUT_FILE, mode='w', index=False)
    else:
        df_batch.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)


# ---------------- ENTRY ----------------

if __name__ == "__main__":
    main()
