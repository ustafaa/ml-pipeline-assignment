import os
import sys

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

EPOCHS = int(os.environ.get("EPOCHS", 5))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.001))
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data()

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("mnist_classifier")

    with mlflow.start_run() as run:
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()

            accuracy = evaluate(model, test_loader, device)
            mlflow.log_metric("accuracy", accuracy, step=epoch)
            print(f"Epoch {epoch}/{EPOCHS} - accuracy: {accuracy:.4f}")

        run_id = run.info.run_id

    with open("model_info.txt", "w") as f:
        f.write(f"{run_id},{accuracy:.4f}")

    print(f"Run ID {run_id} written to model_info.txt")
    return run_id


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"Training failed: {e}", file=sys.stderr)
        sys.exit(1)
