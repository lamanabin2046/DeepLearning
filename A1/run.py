import argparse
import copy
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as tv_models
import matplotlib.pyplot as plt

from alexnet import AlexNet, AlexNetLRN
from googlenet import GoogLeNet, GoogLeNetAux
from resnet import ResNet18


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        return x.transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_ln = self.ln1(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class ViTSmall(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 embed_dim=128, depth=6, n_heads=4, n_classes=10, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(x + self.pos_embed)
        x = self.blocks(x)
        x = self.ln(x[:, 0])
        return self.head(x)


def get_dataloaders(dataset: str, batch_size: int = 64):
    if dataset != "cifar10":
        raise ValueError("Only CIFAR-10 is supported.")

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        ),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=preprocess
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=preprocess
    )

    train_set, val_set = torch.utils.data.random_split(train_dataset, [40000, 10000])

    return {
        "train": torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
        "val": torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2),
        "test": torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2),
    }


def get_cifar10_32_dataloaders(batch_size: int = 64):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        ),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=preprocess
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=preprocess
    )

    train_set, val_set = torch.utils.data.random_split(train_dataset, [40000, 10000])

    return {
        "train": torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
        "val": torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2),
        "test": torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2),
    }


def build_pretrained_alexnet(device):
    model = tv_models.alexnet(weights=tv_models.AlexNet_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(4096, 10)
    return model.to(device)


def build_pretrained_googlenet(device):
    model = tv_models.googlenet(
        weights=tv_models.GoogLeNet_Weights.IMAGENET1K_V1,
        aux_logits=True
    )

    model.fc = nn.Linear(1024, 10)

    if model.aux1 is not None:
        model.aux1.fc2 = nn.Linear(1024, 10)

    if model.aux2 is not None:
        model.aux2.fc2 = nn.Linear(1024, 10)

    return model.to(device)


def build_pretrained_resnet18(device):
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 10)
    return model.to(device)


def build_pretrained_vit_b16(device):
    model = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.DEFAULT)
    model.heads = nn.Linear(768, 10)
    return model.to(device)


def train_model(model, dataloaders, criterion, optimizer, num_epochs, model_name, device):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    val_acc_history = []
    loss_history = []

    os.makedirs("models", exist_ok=True)

    for epoch in range(num_epochs):
        t0 = time.time()

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    if isinstance(outputs, tuple):
                        main_loss = criterion(outputs[0], labels)
                        aux_loss = sum(
                            criterion(output, labels)
                            for output in outputs[1:]
                            if output is not None
                        )
                        loss = main_loss + 0.3 * aux_loss
                        outputs = outputs[0]
                    else:
                        loss = criterion(outputs, labels)

                    preds = outputs.argmax(dim=1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_correct += (preds == labels).sum().item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_correct / len(dataloaders[phase].dataset)

            print(f"  {phase:5s}  loss: {epoch_loss:.4f}  acc: {epoch_acc:.4f}")

            if phase == "val":
                val_acc_history.append(epoch_acc)
                loss_history.append(epoch_loss)

                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                    save_path = os.path.join("models", f"{model_name}_best.pth")
                    torch.save(best_model_wts, save_path)

                    print(f"  --> New best ({best_val_acc:.4f}) saved to {save_path}")

        print(f"  Time: {time.time() - t0:.1f}s")

    print(f"\nBest val accuracy: {best_val_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, val_acc_history, loss_history


def finetune_pretrained(model, dataloaders, model_name, device,
                        stage1_epochs=5, stage2_epochs=10, lr=1e-3):

    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Stage 1: Freeze backbone, train head only ({stage1_epochs} epochs) ---")

    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "fc"):
        model.fc.requires_grad_(True)
        head_params = model.fc.parameters()

    elif hasattr(model, "heads"):
        model.heads.requires_grad_(True)
        head_params = model.heads.parameters()

    elif hasattr(model, "classifier"):
        model.classifier[6].requires_grad_(True)
        head_params = model.classifier[6].parameters()

    else:
        raise AttributeError("Model has no fc, heads, or classifier head.")

    optimizer = optim.Adam(head_params, lr=lr)

    model, val_acc1, loss1 = train_model(
        model, dataloaders, criterion, optimizer,
        stage1_epochs, f"{model_name}_stage1", device
    )

    print(f"\n--- Stage 2: Unfreeze all layers and fine-tune ({stage2_epochs} epochs) ---")

    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=lr * 0.1)

    model, val_acc2, loss2 = train_model(
        model, dataloaders, criterion, optimizer,
        stage2_epochs, f"{model_name}_stage2", device
    )

    return model, val_acc1 + val_acc2, loss1 + loss2


def test_model(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"\nTest accuracy: {acc:.4f}  ({correct}/{total})")

    return acc


def plot_history(val_acc_history, loss_history, model_name):
    os.makedirs("plots", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(loss_history, label="Validation")
    ax1.set_title("Loss per epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(val_acc_history, label="Validation")
    ax2.set_title("Accuracy per epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()

    path = os.path.join("plots", f"{model_name}_curves.png")
    plt.savefig(path)

    print(f"Curves saved to {path}")
    plt.show()


SCRATCH_MODELS = {
    "alexnet": {
        "class": AlexNet,
        "optim": "sgd",
        "lr": 0.001,
        "momentum": 0.9,
        "epochs": 10,
    },
    "alexnet_lrn": {
        "class": AlexNetLRN,
        "optim": "sgd",
        "lr": 0.001,
        "momentum": 0.9,
        "epochs": 10,
    },
    "googlenet": {
        "class": GoogLeNet,
        "optim": "adam",
        "lr": 0.01,
        "epochs": 25,
    },
    "googlenet_aux": {
        "class": GoogLeNetAux,
        "optim": "adam",
        "lr": 0.001,
        "epochs": 25,
    },
    "resnet18": {
        "class": ResNet18,
        "optim": "sgd",
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "epochs": 20,
    },
    "vit_small": {
        "class": ViTSmall,
        "optim": "adam",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 20,
    },
}


PRETRAINED_MODELS = {
    "alexnet_pretrained",
    "googlenet_pretrained",
    "resnet18_pretrained",
    "vit_b16_pretrained",
}


def build_scratch_model(name, device):
    cfg = SCRATCH_MODELS[name]

    model = cfg["class"]().to(device)

    if cfg["optim"] == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg["lr"],
            momentum=cfg.get("momentum", 0.9),
            weight_decay=cfg.get("weight_decay", 0.0),
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 0.0),
        )

    return model, optimizer, cfg["epochs"]


def main():
    parser = argparse.ArgumentParser(description="Train/test CNN and ViT models on CIFAR-10")

    parser.add_argument(
        "--model",
        required=True,
        choices=list(SCRATCH_MODELS.keys()) + list(PRETRAINED_MODELS),
    )
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--weights", type=str, default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device  : {device}")
    print(f"Model   : {args.model}")
    print(f"Dataset : {args.dataset}")

    if args.model in ("googlenet", "resnet18", "vit_small"):
        dataloaders = get_cifar10_32_dataloaders(batch_size=args.batch_size)
    else:
        dataloaders = get_dataloaders(args.dataset, batch_size=args.batch_size)

    if args.train:
        model_name = f"{args.model}_{args.dataset}"

        if args.model in PRETRAINED_MODELS:
            if args.model == "alexnet_pretrained":
                model = build_pretrained_alexnet(device)
            elif args.model == "googlenet_pretrained":
                model = build_pretrained_googlenet(device)
            elif args.model == "resnet18_pretrained":
                model = build_pretrained_resnet18(device)
            elif args.model == "vit_b16_pretrained":
                model = build_pretrained_vit_b16(device)
            else:
                raise ValueError(f"Unknown pretrained model: {args.model}")

            total_epochs = args.epochs if args.epochs else 15
            stage1_epochs = max(1, total_epochs // 3)
            stage2_epochs = total_epochs - stage1_epochs

            n_params = sum(p.numel() for p in model.parameters())

            print(f"Params  : {n_params:,}")
            print(f"Epochs  : {total_epochs}")
            print(f"Stage 1 : {stage1_epochs} epochs")
            print(f"Stage 2 : {stage2_epochs} epochs")

            model, val_acc_history, loss_history = finetune_pretrained(
                model,
                dataloaders,
                model_name,
                device,
                stage1_epochs=stage1_epochs,
                stage2_epochs=stage2_epochs,
            )

        else:
            model, optimizer, default_epochs = build_scratch_model(args.model, device)

            num_epochs = args.epochs if args.epochs else default_epochs
            n_params = sum(p.numel() for p in model.parameters())

            print(f"Params  : {n_params:,}")
            print(f"Epochs  : {num_epochs}")

            criterion = nn.CrossEntropyLoss()

            model, val_acc_history, loss_history = train_model(
                model,
                dataloaders,
                criterion,
                optimizer,
                num_epochs,
                model_name,
                device,
            )

        plot_history(val_acc_history, loss_history, model_name)

    if args.test:
        if args.model in PRETRAINED_MODELS:
            if args.model == "alexnet_pretrained":
                model = build_pretrained_alexnet(device)
            elif args.model == "googlenet_pretrained":
                model = build_pretrained_googlenet(device)
            elif args.model == "resnet18_pretrained":
                model = build_pretrained_resnet18(device)
            elif args.model == "vit_b16_pretrained":
                model = build_pretrained_vit_b16(device)
            else:
                raise ValueError(f"Unknown pretrained model: {args.model}")
        else:
            model, _, _ = build_scratch_model(args.model, device)

        weights_path = args.weights or os.path.join(
            "models",
            f"{args.model}_{args.dataset}_best.pth",
        )

        if not os.path.exists(weights_path):
            print(f"Weights file not found: {weights_path}")
            return

        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")

        test_model(model, dataloaders["test"], device)

    if not args.train and not args.test:
        parser.print_help()


if __name__ == "__main__":
    main()