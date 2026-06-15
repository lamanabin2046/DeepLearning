from __future__ import division
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time
from tqdm import tqdm

# ─── Constants ───────────────────────────────────────────────
IMG_SIZE   = 128
N_CLASSES  = 3
BATCH_SIZE = 16
EPOCHS     = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ─── Dataset ──────────────────────────────────────────────────
from torchvision.datasets import OxfordIIITPet

class PetSegDataset(Dataset):
    def __init__(self, base, size=128):
        self.ds = base
        self.img_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((size, size), 
                interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor(),
        ])

    def __len__(self): 
        return len(self.ds)

    def __getitem__(self, idx):
        img, mask = self.ds[idx]
        img  = self.img_tf(img)
        mask = (self.mask_tf(mask).squeeze(0).long() - 1).clamp(0, 2)
        return img, mask


def get_dataloaders():
    os.makedirs('./data', exist_ok=True)
    train_raw = OxfordIIITPet('./data', split='trainval', 
                               target_types='segmentation', download=True)
    test_raw  = OxfordIIITPet('./data', split='test',     
                               target_types='segmentation', download=True)
    train_data   = PetSegDataset(train_raw, IMG_SIZE)
    test_data    = PetSegDataset(test_raw,  IMG_SIZE)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, 
                              shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, 
                              shuffle=False, num_workers=2)
    print(f'Train: {len(train_data)} | Test: {len(test_data)}')
    return train_loader, test_loader, test_data



# ─── Models ───────────────────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


# ─── UNet with ResNet-18 encoder + skip connections ───────────
class UNet_ResNet18(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        resnet = models.resnet18(weights='IMAGENET1K_V1')

        # Encoder stages from ResNet-18
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64ch, /2
        self.pool = resnet.maxpool                                          # /4
        self.enc1 = resnet.layer1   # 64ch,  /4
        self.enc2 = resnet.layer2   # 128ch, /8
        self.enc3 = resnet.layer3   # 256ch, /16
        self.enc4 = resnet.layer4   # 512ch, /32

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(256+256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(128+128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(64+64, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = DoubleConv(64+64, 64)

        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = DoubleConv(32, 32)

        self.output = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)       # 64ch, H/2
        e1 = self.enc1(self.pool(e0))  # 64ch, H/4
        e2 = self.enc2(e1)      # 128ch, H/8
        e3 = self.enc3(e2)      # 256ch, H/16
        e4 = self.enc4(e3)      # 512ch, H/32

        # Decoder with skip connections
        d4 = self.up4(e4)
        if d4.shape != e3.shape: d4 = F.interpolate(d4, size=e3.shape[2:])
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = self.up3(d4)
        if d3.shape != e2.shape: d3 = F.interpolate(d3, size=e2.shape[2:])
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)
        if d2.shape != e1.shape: d2 = F.interpolate(d2, size=e1.shape[2:])
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        d1 = self.up1(d2)
        if d1.shape != e0.shape: d1 = F.interpolate(d1, size=e0.shape[2:])
        d1 = self.dec1(torch.cat([d1, e0], dim=1))

        d0 = self.up0(d1)
        d0 = self.dec0(d0)

        return self.output(d0)


# ─── UNet with ResNet-18 encoder WITHOUT skip connections ──────
class UNet_ResNet18_NoSkip(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        resnet = models.resnet18(weights='IMAGENET1K_V1')

        # Same encoder
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4

        # Decoder WITHOUT skip connections
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(64, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = DoubleConv(64, 64)

        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = DoubleConv(32, 32)

        self.output = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Decoder WITHOUT skip connections
        d4 = self.dec4(self.up4(e4))
        d3 = self.dec3(self.up3(d4))
        d2 = self.dec2(self.up2(d3))
        d1 = self.dec1(self.up1(d2))
        d0 = self.dec0(self.up0(d1))

        return self.output(d0)


def get_model(model_name):
    if model_name == 'unet_resnet18':
        return UNet_ResNet18(n_classes=N_CLASSES)
    elif model_name == 'unet_resnet18_no_skip':
        return UNet_ResNet18_NoSkip(n_classes=N_CLASSES)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ─── Metrics ──────────────────────────────────────────────────
def compute_iou(pred, target, n_classes=3):
    pred = pred.argmax(dim=1)
    ious = []
    for cls in range(n_classes):
        inter = ((pred==cls) & (target==cls)).sum().float()
        union = ((pred==cls) | (target==cls)).sum().float()
        if union > 0:
            ious.append((inter/union).item())
    return np.mean(ious) if ious else 0.0


# ─── Training Function ────────────────────────────────────────
def train_model(model, train_loader, test_loader, epochs, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses, val_ious = [], []
    best_miou = 0.0

    for epoch in range(epochs):
        # Training
        model.train()
        ep_loss = []
        start_time = time.time()
        for imgs, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            imgs, masks = imgs.to(device), masks.to(device)
            loss = criterion(model(imgs), masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss.append(loss.item())

        # Evaluation
        model.eval()
        ep_iou = []
        with torch.no_grad():
            for imgs, masks in test_loader:
                ep_iou.append(compute_iou(model(imgs.to(device)), masks.to(device)))

        scheduler.step()
        epoch_time = time.time() - start_time
        train_losses.append(np.mean(ep_loss))
        val_ious.append(np.mean(ep_iou))

        print(f'Epoch {epoch+1:02d} | Loss: {train_losses[-1]:.4f} | '
              f'mIoU: {val_ious[-1]:.4f} | Time: {epoch_time:.1f}s')

        # Save best model
        if val_ious[-1] > best_miou:
            best_miou = val_ious[-1]
            torch.save(model.state_dict(), f'{model_name}.pt')
            print(f'  → Best model saved! mIoU: {best_miou:.4f}')

    print(f'\nBest mIoU: {best_miou:.4f}')
    return train_losses, val_ious


# ─── Evaluation Function ──────────────────────────────────────
def evaluate_model(model, test_loader):
    model.eval()
    ep_iou = []
    with torch.no_grad():
        for imgs, masks in test_loader:
            ep_iou.append(compute_iou(model(imgs.to(device)), masks.to(device)))
    miou = np.mean(ep_iou)
    print(f'mIoU: {miou:.4f}')
    return miou


# ─── Visualization Function ───────────────────────────────────
def visualize(model, test_data, model_name):
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    CLASS_COLORS = np.array([[255,100,100],[100,100,255],[255,255,100]], dtype=np.uint8)
    CLASS_NAMES  = ['Pet', 'Background', 'Border']

    model.eval()
    fig, axes = plt.subplots(5, 4, figsize=(14, 18))
    for ax, t in zip(axes[0], ['Input','Ground Truth','Prediction','Overlay']):
        ax.set_title(t, fontsize=11, fontweight='bold')

    for row in range(5):
        img, mask = test_data[row*50]
        with torch.no_grad():
            pred_mask = model(img.unsqueeze(0).to(device)).argmax(1).squeeze().cpu().numpy()
        img_d = torch.clamp(img * std + mean, 0, 1).permute(1,2,0).numpy()
        axes[row][0].imshow(img_d)
        axes[row][1].imshow(CLASS_COLORS[mask.numpy()])
        axes[row][2].imshow(CLASS_COLORS[pred_mask])
        axes[row][3].imshow(img_d)
        axes[row][3].imshow(CLASS_COLORS[pred_mask], alpha=0.5)
        for ax in axes[row]: ax.axis('off')

    patches = [plt.Rectangle((0,0),1,1,color=CLASS_COLORS[i]/255) for i in range(3)]
    fig.legend(patches, CLASS_NAMES, loc='lower center', ncol=3)
    plt.suptitle(f'U-Net Results — {model_name}', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{model_name}_results.png')
    print(f'Visualization saved to {model_name}_results.png')


# ─── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="unet_resnet18",
                        choices=["unet_resnet18", "unet_resnet18_no_skip"])
    parser.add_argument("--dataset", default="oxford_pet")
    parser.add_argument("--epochs",  type=int, default=20)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    # Load data
    train_loader, test_loader, test_data = get_dataloaders()

    # Load model
    model = get_model(args.model).to(device)
    print(f'Model: {args.model}')
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

    if args.train:
        print(f'\nTraining {args.model} for {args.epochs} epochs...')
        train_losses, val_ious = train_model(
            model, train_loader, test_loader, args.epochs, args.model
        )
        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(train_losses, marker='o', color='steelblue')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].grid(True)
        axes[1].plot(val_ious, marker='s', color='darkorange')
        axes[1].set_title('Validation mIoU')
        axes[1].set_xlabel('Epoch')
        axes[1].grid(True)
        plt.tight_layout()
        plt.savefig(f'{args.model}_training.png')
        print(f'Training plot saved!')

        # Visualize predictions
        model.load_state_dict(torch.load(f'{args.model}.pt', map_location=device))
        visualize(model, test_data, args.model)

    elif args.evaluate:
        if args.weights is None:
            args.weights = f'{args.model}.pt'
        print(f'\nEvaluating {args.model} from {args.weights}...')
        model.load_state_dict(torch.load(args.weights, map_location=device))
        evaluate_model(model, test_loader)
        visualize(model, test_data, args.model)