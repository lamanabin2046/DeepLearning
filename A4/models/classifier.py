"""
Lightweight CNN classifier for MNIST digits (0-9).

This is NOT part of the generative models themselves -- it's a helper
"judge" used in Exercise 1 to automatically label what digit each
GAN-generated image looks like, so we can build the mode-collapse
histogram (Count out of 1000 per digit).

Train once with:
    python run.py --train-classifier

This saves saved/mnist_classifier.pt, which exercise1_mode_collapse.py
then loads to classify generated samples.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTClassifier(nn.Module):
    """Small CNN, >99% test accuracy on MNIST after a few epochs."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # x: (B, 1, 28, 28), expected in range roughly [-1, 1] or [0, 1]
        x = self.pool(F.relu(self.conv1(x)))   # -> (B, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # -> (B, 64, 7, 7)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # logits


def train_classifier(device, data_root='./data', epochs=3, batch_size=256,
                      save_path='saved/mnist_classifier.pt'):
    """Train the helper classifier on real MNIST (separate from the GAN)."""
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    # NOTE: classifier expects the SAME normalization as the GAN's output
    # ([-1, 1]) so that classifying generated images works correctly.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_set = torchvision.datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    model = MNISTClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        print(f'[classifier] epoch {epoch+1}/{epochs} | test acc: {correct/total:.4f}')

    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f'[classifier] saved to {save_path}')
    return model


def load_classifier(device, path='saved/mnist_classifier.pt'):
    model = MNISTClassifier().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model