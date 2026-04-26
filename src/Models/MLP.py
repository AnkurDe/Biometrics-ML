from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# -----------------------------
# Load Data
# -----------------------------
path = Path.cwd().parent.parent / "TabulatedData" / "values_all_layers_16bit.parquet"

dataset = pd.read_parquet(path)

X = dataset.drop("name", axis=1).values
y = dataset["name"].values

le = LabelEncoder()
y = le.fit_transform(y)

print("Data Loaded")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

del dataset, X, y

# -----------------------------
# Feature Scaling (recommended for MLP)
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Convert to PyTorch tensors
# -----------------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# -----------------------------
# GPU Support
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

# -----------------------------
# Define Model
# Equivalent to:
# hidden_layer_sizes=(2,)
# activation='logistic'
# -----------------------------
input_size = X_train.shape[1]
num_classes = len(le.classes_)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.Sigmoid(),          # logistic activation
            nn.Linear(100, 101),
            nn.Sigmoid(),
            nn.Linear(101, 100),
            nn.Sigmoid(),
            nn.Linear(100, 100),
            nn.Sigmoid(),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = MLP().to(device)

# -----------------------------
# Loss and Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Training Loop
# -----------------------------
epochs = 1000

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()

        outputs = model(xb)
        loss = criterion(outputs, yb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

# -----------------------------
# Evaluation
# -----------------------------
model.eval()

with torch.no_grad():
    outputs = model(X_test)
    preds = torch.argmax(outputs, dim=1)

acc = accuracy_score(y_test.cpu().numpy(), preds.cpu().numpy())

print(f"\nTest Accuracy: {acc:.4f}")
print(classification_report(y_test.cpu().numpy(), preds.cpu().numpy(), target_names=le.classes_, zero_division=0))
