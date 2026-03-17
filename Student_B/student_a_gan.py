import pandas as pd
import torch
import torch.nn as nn


print("Loading MNIST CSV data...")
df = pd.read_csv('student_A/mnist_test.csv')

# Drop (column 0) to keep pixel data
pixel_data = df.iloc[:, 1:].values / 255.0
data = torch.tensor(pixel_data, dtype=torch.float32)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(10, 128), nn.ReLU(), nn.Linear(128, 784), nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


G = Generator()
D = Discriminator()
opt_G = torch.optim.Adam(G.parameters(), lr=0.001)
opt_D = torch.optim.Adam(D.parameters(), lr=0.001)
criterion = nn.BCELoss()

print("Training GAN...")
for epoch in range(5):
    indices = torch.randint(0, len(data), (16,))
    real_data = data[indices]

    # Discriminator
    opt_D.zero_grad()
    noise = torch.randn(16, 10)
    fake_data = G(noise).detach()
    loss_D = criterion(D(real_data), torch.ones(16, 1)) + criterion(D(fake_data), torch.zeros(16, 1))
    loss_D.backward()
    opt_D.step()

    # Generator
    opt_G.zero_grad()
    noise = torch.randn(16, 10)
    fake_data = G(noise)
    loss_G = criterion(D(fake_data), torch.ones(16, 1))
    loss_G.backward()
    opt_G.step()

print(f"Training complete. Generator Final Loss: {loss_G.item()}")
