from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from torchvision.models import AlexNet_Weights
from PIL import Image
import pandas as pd
from tqdm import tqdm

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


# ---------------- Preprocessing ----------------

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ---------------- Dataset ----------------

class ImageDataset(Dataset):
    def __init__(self, root_path):
        self.samples = []

        for d in root_path.iterdir():
            if d.is_dir():
                for f in d.iterdir():
                    if f.is_file():
                        self.samples.append((f, d.name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file, label = self.samples[idx]

        try:
            with Image.open(file) as img:
                img = img.convert("RGB")
                tensor = preprocess(img)
        except Exception as e:
            print(f"Skipping {file}: {e}")
            return None

        return tensor, label


# ---------------- Hook Storage ----------------

layer_outputs = {}

def get_hook(name):
    def hook(module, input, output):
        layer_outputs[name] = output.detach()
    return hook


# Register hooks
for name, layer in alexnet.features._modules.items():
    layer.register_forward_hook(get_hook(f"features_{name}"))

alexnet.avgpool.register_forward_hook(get_hook("avgpool"))

for name, layer in alexnet.classifier._modules.items():
    layer.register_forward_hook(get_hook(f"classifier_{name}"))


# ---------------- Feature Extraction ----------------

def extract_features_batch(inputs):
    global layer_outputs
    layer_outputs = {}

    with torch.no_grad():
        _ = alexnet(inputs)

    batch_results = []

    batch_size = inputs.shape[0]

    for i in range(batch_size):
        result = {}

        for layer_name, output in layer_outputs.items():

            if len(output.shape) == 4:
                flat = output[i].mean(dim=[1, 2]).view(-1).cpu().numpy()
            else:
                flat = output[i].view(-1).cpu().numpy()

            for j, v in enumerate(flat):
                result[f"{layer_name}_v{j+1}"] = float(v)

        batch_results.append(result)

    return batch_results


# ---------------- Collate Function ----------------

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None

    images, labels = zip(*batch)
    return torch.stack(images), labels


# ---------------- CSV Writer ----------------

def write_batch(rows, header_written, output_file):
    df = pd.DataFrame(rows)

    if not header_written:
        df.to_csv(output_file, mode='w', index=False)
    else:
        df.to_csv(output_file, mode='a', header=False, index=False)


# ---------------- MAIN ----------------

def main():

    DATA_PATH = Path("../Processed_Data")
    OUTPUT_FILE = "values_all_layers_t_dl.csv"

    dataset = ImageDataset(DATA_PATH)

    dataloader = DataLoader(
        dataset,
        batch_size=100,          # tune based on GPU
        shuffle=False,
        num_workers=4,          # increase for faster disk loading
        pin_memory=True,
        collate_fn=collate_fn
    )

    header_written = False

    progress = tqdm(dataloader, desc="Extracting", unit="batch")

    for images, labels in progress:

        if images is None:
            continue

        images = images.to(device)

        features_batch = extract_features_batch(images)

        rows = []
        for label, features in zip(labels, features_batch):
            row = {"name": label}
            row.update(features)
            rows.append(row)

        write_batch(rows, header_written, OUTPUT_FILE)
        header_written = True

        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**2
            progress.set_postfix_str(f"VRAM {allocated:.0f} MB")
            torch.cuda.empty_cache()

    print(f"{OUTPUT_FILE} written successfully")


# ---------------- ENTRY ----------------

if __name__ == "__main__":
    main()
