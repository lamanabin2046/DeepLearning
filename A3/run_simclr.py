import os
import time
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10


def make_dirs():
    os.makedirs("saved", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_curve(values, title, ylabel, save_path):
    plt.figure(figsize=(7, 4))
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


class CIFAR10Pair(CIFAR10):
    def __init__(self, root="./data", train=True, download=True):
        self.transform_pair = T.Compose([
            T.RandomResizedCrop(32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2470, 0.2435, 0.2616))
        ])
        super().__init__(root=root, train=train, download=download, transform=None)

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        from PIL import Image
        img = Image.fromarray(img).convert("RGB")
        x1 = self.transform_pair(img)
        x2 = self.transform_pair(img)
        return x1, x2, label


class TinyViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3,
                 embed_dim=192, depth=4, num_heads=3):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        x = self.encoder(x)
        x = self.norm(x)

        return x[:, 0]


class SimCLR(nn.Module):
    def __init__(self, embed_dim=192, proj_dim=128):
        super().__init__()
        self.encoder = TinyViT(embed_dim=embed_dim)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim)
        )

    def forward(self, x):
        feat = self.encoder(x)
        z = self.projector(feat)
        z = F.normalize(z, dim=1)
        return z

    def features(self, x):
        return self.encoder(x)


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)              # [2B, D]
        sim = torch.matmul(z, z.T) / self.temperature

        mask = torch.eye(2 * B, device=z.device).bool()
        sim = sim.masked_fill(mask, -1e9)

        labels = torch.arange(2 * B, device=z.device)
        labels = (labels + B) % (2 * B)

        loss = F.cross_entropy(sim, labels)
        return loss


def get_eval_loaders(batch_size=256):
    train_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616))
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616))
    ])

    train_set = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_set = CIFAR10(root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    return train_loader, test_loader


def train_simclr(args):
    make_dirs()
    set_seed(args.seed)
    device = get_device()

    print("=" * 60)
    print("Training SimCLR")
    print("Device:", device)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Temperature:", args.temperature)
    print("=" * 60)

    dataset = CIFAR10Pair(root="./data", train=True, download=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    model = SimCLR().to(device)
    criterion = NTXentLoss(temperature=args.temperature)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    losses = []
    epoch_times = []

    for epoch in range(args.epochs):
        model.train()
        ep_losses = []
        start = time.time()

        for x1, x2, _ in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            z1 = model(x1)
            z2 = model(x2)

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()

            ep_losses.append(loss.item())

        elapsed = time.time() - start
        avg_loss = float(np.mean(ep_losses))
        losses.append(avg_loss)
        epoch_times.append(elapsed)

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"Contrastive Loss: {avg_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

    save_path = "saved/simclr.pt"
    torch.save({
        "model": model.state_dict(),
        "losses": losses,
        "epoch_times": epoch_times,
        "args": vars(args)
    }, save_path)

    print("Saved model:", save_path)

    plot_curve(
        losses,
        title="SimCLR Contrastive Loss",
        ylabel="Contrastive Loss",
        save_path="results/simclr_loss.png"
    )

    print("Saved plot: results/simclr_loss.png")


def linear_eval_simclr(args):
    make_dirs()
    set_seed(args.seed)
    device = get_device()

    print("=" * 60)
    print("SimCLR Linear Evaluation")
    print("Weights:", args.weights)
    print("Device:", device)
    print("=" * 60)

    model = SimCLR().to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    classifier = nn.Linear(192, 10).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = get_eval_loaders(batch_size=args.batch_size)

    test_accs = []

    for epoch in range(args.linear_epochs):
        classifier.train()
        correct = 0
        total = 0
        losses = []

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                feats = model.features(images)

            logits = classifier(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_acc = 100.0 * correct / total

        classifier.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                feats = model.features(images)
                logits = classifier(feats)

                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

        test_acc = 100.0 * correct / total
        test_accs.append(test_acc)

        print(
            f"Linear Epoch {epoch+1:03d}/{args.linear_epochs} | "
            f"Loss: {np.mean(losses):.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}%"
        )

    result_path = "results/simclr_linear.txt"

    with open(result_path, "w") as f:
        f.write(f"Weights: {args.weights}\n")
        f.write(f"Final Linear Eval Accuracy: {test_accs[-1]:.2f}%\n")
        f.write(f"Best Linear Eval Accuracy: {max(test_accs):.2f}%\n")

    print("Final Linear Eval Accuracy:", f"{test_accs[-1]:.2f}%")
    print("Best Linear Eval Accuracy:", f"{max(test_accs):.2f}%")
    print("Saved result:", result_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--linear", action="store_true")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--linear-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--weights", type=str, default="saved/simclr.pt")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.train:
        train_simclr(args)

    elif args.evaluate and args.linear:
        linear_eval_simclr(args)

    else:
        print("Please choose one mode:")
        print("Train SimCLR:")
        print("  python run_simclr.py --epochs 50 --batch-size 256 --train")
        print("Linear eval SimCLR:")
        print("  python run_simclr.py --weights saved/simclr.pt --evaluate --linear --linear-epochs 20")


if __name__ == "__main__":
    main()