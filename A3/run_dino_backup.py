
import os
import time
import argparse
import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Utility
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dirs():
    os.makedirs("saved", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)


def make_tag(args):
    if args.no_centering:
        return "dino_no_centering"
    if args.n_local == 0:
        return "dino_no_local"
    return "dino_default"


def plot_curve(values, title, ylabel, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(values, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -----------------------------
# Tiny Vision Transformer
# -----------------------------
class TinyViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3,
                 embed_dim=192, depth=4, num_heads=3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: [B, 3, 32, 32]
        x = self.patch_embed(x)          # [B, 192, 8, 8]
        x = x.flatten(2).transpose(1, 2) # [B, 64, 192]

        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, 192]
        x = torch.cat([cls, x], dim=1)          # [B, 65, 192]
        x = x + self.pos_embed

        x = self.encoder(x)
        x = self.norm(x)

        cls_feature = x[:, 0]  # [B, 192]
        return cls_feature


# -----------------------------
# DINO Projection Head
# -----------------------------
class DINOHead(nn.Module):
    def __init__(self, in_dim=192, hidden_dim=512, out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return x


class DINOModel(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.backbone = TinyViT()
        self.head = DINOHead(in_dim=192, out_dim=out_dim)

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        return out

    def features(self, x):
        return self.backbone(x)


# -----------------------------
# DINO Loss
# -----------------------------
class DINOLoss(nn.Module):
    def __init__(self, out_dim=256, teacher_temp=0.04, student_temp=0.1,
                 center_momentum=0.9, use_centering=True):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.use_centering = use_centering
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out):
        # student_out: list of outputs from all crops
        # teacher_out: list of outputs from 2 global crops

        s_probs = [
            F.log_softmax(s / self.student_temp, dim=-1)
            for s in student_out
        ]

        if self.use_centering:
            t_probs = [
                F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach()
                for t in teacher_out
            ]
        else:
            t_probs = [
                F.softmax(t / self.teacher_temp, dim=-1).detach()
                for t in teacher_out
            ]

        total_loss = 0.0
        n_terms = 0

        for t_idx, t_prob in enumerate(t_probs):
            for s_idx, s_log_prob in enumerate(s_probs):
                # skip same global view
                if s_idx == t_idx:
                    continue

                loss = -(t_prob * s_log_prob).sum(dim=-1).mean()
                total_loss += loss
                n_terms += 1

        total_loss = total_loss / n_terms

        if self.use_centering:
            self.update_center(torch.stack(teacher_out).mean(dim=0))

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_mean):
        batch_center = teacher_mean.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


# -----------------------------
# CIFAR-10 Multi-crop Dataset
# -----------------------------
class CIFAR10DINO(CIFAR10):
    def __init__(self, root="./data", train=True, download=True, n_local=4):
        super().__init__(root=root, train=train, download=download)
        self.n_local = n_local

        self.global_transform = T.Compose([
            T.RandomResizedCrop(32, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2470, 0.2435, 0.2616))
        ])

        self.local_transform = T.Compose([
            T.RandomResizedCrop(32, scale=(0.2, 0.5)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2470, 0.2435, 0.2616))
        ])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        from PIL import Image
        img = Image.fromarray(img).convert("RGB")

        crops = []
        crops.append(self.global_transform(img))
        crops.append(self.global_transform(img))

        for _ in range(self.n_local):
            crops.append(self.local_transform(img))

        return crops, target


def get_eval_loaders(batch_size=256):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616))
    ])

    train_set = CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_set = CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    return train_loader, test_loader


# -----------------------------
# Teacher EMA update
# -----------------------------
@torch.no_grad()
def update_teacher(student, teacher, momentum=0.996):
    for s_param, t_param in zip(student.parameters(), teacher.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1 - momentum)


# -----------------------------
# Train DINO
# -----------------------------
def train_dino(args):
    make_dirs()
    set_seed(args.seed)
    device = get_device()

    tag = make_tag(args)
    print("=" * 60)
    print("Training DINO")
    print("Setting:", tag)
    print("Device:", device)
    print("Epochs:", args.epochs)
    print("Local crops:", args.n_local)
    print("Use centering:", not args.no_centering)
    print("=" * 60)

    dataset = CIFAR10DINO(
        root="./data",
        train=True,
        download=True,
        n_local=args.n_local
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    student = DINOModel(out_dim=args.out_dim).to(device)
    teacher = copy.deepcopy(student).to(device)

    for p in teacher.parameters():
        p.requires_grad = False

    criterion = DINOLoss(
        out_dim=args.out_dim,
        use_centering=not args.no_centering
    ).to(device)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    all_losses = []
    center_norms = []
    epoch_times = []

    for epoch in range(args.epochs):
        student.train()
        teacher.eval()

        epoch_losses = []
        start = time.time()

        for crops, _ in loader:
            crops = [c.to(device, non_blocking=True) for c in crops]

            # teacher sees only 2 global crops
            with torch.no_grad():
                teacher_out = [teacher(crops[0]), teacher(crops[1])]

            # student sees all crops
            student_out = [student(c) for c in crops]

            loss = criterion(student_out, teacher_out)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            optimizer.step()

            update_teacher(student, teacher, momentum=args.teacher_momentum)

            epoch_losses.append(loss.item())

        elapsed = time.time() - start
        avg_loss = float(np.mean(epoch_losses))
        center_norm = float(criterion.center.norm().item())

        all_losses.append(avg_loss)
        center_norms.append(center_norm)
        epoch_times.append(elapsed)

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Center norm: {center_norm:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

    save_path = f"saved/{tag}.pt"

    torch.save({
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "losses": all_losses,
        "center_norms": center_norms,
        "epoch_times": epoch_times,
        "args": vars(args)
    }, save_path)

    print("Saved model:", save_path)

    plot_curve(
        all_losses,
        title=f"{tag} Training Loss",
        ylabel="Loss",
        save_path=f"results/{tag}_loss.png"
    )

    plot_curve(
        center_norms,
        title=f"{tag} Center Norm",
        ylabel="Center Norm",
        save_path=f"results/{tag}_center_norm.png"
    )

    print("Saved plots:")
    print(f"results/{tag}_loss.png")
    print(f"results/{tag}_center_norm.png")


# -----------------------------
# Linear Evaluation
# -----------------------------
def linear_eval(args):
    make_dirs()
    set_seed(args.seed)
    device = get_device()

    if args.weights is None:
        raise ValueError("Please provide --weights saved/model_name.pt")

    print("=" * 60)
    print("Linear Evaluation")
    print("Weights:", args.weights)
    print("Device:", device)
    print("=" * 60)

    model = DINOModel(out_dim=args.out_dim).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["student"])
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    classifier = nn.Linear(192, 10).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = get_eval_loaders(batch_size=args.batch_size)

    train_accs = []
    test_accs = []

    for epoch in range(args.linear_epochs):
        classifier.train()
        model.eval()

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

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(
            f"Linear Epoch {epoch+1:03d}/{args.linear_epochs} | "
            f"Loss: {np.mean(losses):.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}%"
        )

    result_path = args.weights.replace("saved/", "results/").replace(".pt", "_linear.txt")

    with open(result_path, "w") as f:
        f.write(f"Weights: {args.weights}\n")
        f.write(f"Final Linear Eval Accuracy: {test_accs[-1]:.2f}%\n")
        f.write(f"Best Linear Eval Accuracy: {max(test_accs):.2f}%\n")

    print("Final Linear Eval Accuracy:", f"{test_accs[-1]:.2f}%")
    print("Best Linear Eval Accuracy:", f"{max(test_accs):.2f}%")
    print("Saved result:", result_path)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="dino", choices=["dino"])
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--linear", action="store_true")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--linear-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--n-local", type=int, default=4)
    parser.add_argument("--no-centering", action="store_true")

    parser.add_argument("--out-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--teacher-momentum", type=float, default=0.996)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.model == "dino" and args.train:
        train_dino(args)

    elif args.model == "dino" and args.evaluate and args.linear:
        linear_eval(args)

    else:
        print("Please choose one mode:")
        print("Train DINO:")
        print("  python run.py --model dino --epochs 50 --train")
        print("Linear eval:")
        print("  python run.py --model dino --weights saved/dino_default.pt --evaluate --linear")


if __name__ == "__main__":
    main()
