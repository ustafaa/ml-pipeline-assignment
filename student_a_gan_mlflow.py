
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ──────────────────────────────────────────────
# 1.  HYPERPARAMETER CONFIGURATIONS (5 runs)
# ──────────────────────────────────────────────
RUN_CONFIGS = [
    {"learning_rate": 0.0002, "batch_size": 64, "epochs": 50, "latent_dim": 64, "hidden_dim": 256, "tag": "baseline"},
    {"learning_rate": 0.001, "batch_size": 64, "epochs": 50, "latent_dim": 64, "hidden_dim": 256, "tag": "high_lr"},
    {"learning_rate": 0.0001, "batch_size": 64, "epochs": 50, "latent_dim": 64, "hidden_dim": 256, "tag": "low_lr"},
    {"learning_rate": 0.0002, "batch_size": 128, "epochs": 50, "latent_dim": 64, "hidden_dim": 256, "tag": "large_batch"},
    {"learning_rate": 0.0002, "batch_size": 64, "epochs": 50, "latent_dim": 128, "hidden_dim": 512, "tag": "bigger_model"},
]

# ──────────────────────────────────────────────
# 2.  MODEL DEFINITIONS
# ──────────────────────────────────────────────


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# ──────────────────────────────────────────────
# 3.  DATA LOADING  (torchvision MNIST)
# ──────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),           # [0, 1] range, shape (1, 28, 28)
    transforms.Lambda(lambda x: x.view(-1)),  # flatten to 784
])

print("Downloading / loading MNIST via torchvision …")
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)


# ──────────────────────────────────────────────
# 4.  TRAINING FUNCTION (single run)
# ──────────────────────────────────────────────
def train_gan(config: dict, run_number: int):
    """Train one GAN configuration and log everything to MLflow."""

    lr = config["learning_rate"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    latent_dim = config["latent_dim"]
    hidden_dim = config["hidden_dim"]
    run_tag = config["tag"]

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    G = Generator(latent_dim=latent_dim, hidden_dim=hidden_dim)
    D = Discriminator(hidden_dim=hidden_dim)

    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # ── MLflow Run ──────────────────────────
    with mlflow.start_run(run_name=f"run_{run_number}_{run_tag}"):

        # Log parameters
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("latent_dim", latent_dim)
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("beta1", 0.5)

        # Log tags
        mlflow.set_tag("student_id", "YOUR_ID")      # ← replace with your ID
        mlflow.set_tag("run_description", run_tag)
        mlflow.set_tag("model_type", "GAN")

        print(f"\n{'='*60}")
        print(f"  Run {run_number}: {run_tag}  |  lr={lr}  bs={batch_size}  "
              f"latent={latent_dim}  hidden={hidden_dim}")
        print(f"{'='*60}")

        for epoch in range(1, epochs + 1):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            epoch_d_real_correct = 0
            epoch_d_fake_correct = 0
            num_batches = 0

            for real_data, _ in dataloader:
                current_bs = real_data.size(0)

                # Labels
                real_labels = torch.ones(current_bs, 1)
                fake_labels = torch.zeros(current_bs, 1)

                # ── Train Discriminator ──
                opt_D.zero_grad()
                noise = torch.randn(current_bs, latent_dim)
                fake_data = G(noise).detach()

                d_real_out = D(real_data)
                d_fake_out = D(fake_data)

                loss_D = criterion(d_real_out, real_labels) + criterion(d_fake_out, fake_labels)
                loss_D.backward()
                opt_D.step()

                # ── Train Generator ──
                opt_G.zero_grad()
                noise = torch.randn(current_bs, latent_dim)
                fake_data = G(noise)
                d_out = D(fake_data)

                loss_G = criterion(d_out, real_labels)
                loss_G.backward()
                opt_G.step()

                # ── Discriminator "accuracy" ──
                epoch_d_real_correct += (d_real_out > 0.5).sum().item()
                epoch_d_fake_correct += (d_fake_out < 0.5).sum().item()
                epoch_d_loss += loss_D.item()
                epoch_g_loss += loss_G.item()
                num_batches += 1

            # ── Epoch averages ──
            avg_d_loss = epoch_d_loss / num_batches
            avg_g_loss = epoch_g_loss / num_batches
            total_samples = num_batches * batch_size
            d_accuracy = (epoch_d_real_correct + epoch_d_fake_correct) / (2 * total_samples)

            # ── Live Logging (every epoch) ──
            mlflow.log_metric("d_loss", avg_d_loss, step=epoch)
            mlflow.log_metric("g_loss", avg_g_loss, step=epoch)
            mlflow.log_metric("d_accuracy", d_accuracy, step=epoch)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{epochs}  |  D_loss={avg_d_loss:.4f}  "
                      f"G_loss={avg_g_loss:.4f}  D_acc={d_accuracy:.4f}")

        # ── Log final metrics as summary ──
        mlflow.log_metric("final_d_loss", avg_d_loss)
        mlflow.log_metric("final_g_loss", avg_g_loss)
        mlflow.log_metric("final_d_accuracy", d_accuracy)

        # ── Save model artifact (MLflow PyTorch flavor) ──
        mlflow.pytorch.log_model(G, artifact_path="generator_model")
        mlflow.pytorch.log_model(D, artifact_path="discriminator_model")

        print(f"  ✓ Run {run_number} complete — models logged to MLflow.\n")


if __name__ == "__main__":
    mlflow.set_experiment("Mlflow_exp_1")

    for i, cfg in enumerate(RUN_CONFIGS, start=1):
        train_gan(cfg, run_number=i)
