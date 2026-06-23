"""
A4 Generative Models -- main entrypoint.

Usage examples (matching the assignment's required commands):

    # Train Vanilla GAN on MNIST
    python run.py --model gan --dataset mnist --epochs 20 --train

    # Exercise 1(b): intentionally cause mode collapse via a 3x discriminator LR
    python run.py --model gan --dataset mnist --epochs 20 --train --d_lr 6e-4 \
                   --save-name gan_mnist_collapsed.pt

    # Train the helper MNIST classifier (used to judge generated digits)
    python run.py --train-classifier --epochs 3

    # CycleGAN / DDPM commands are added in later exercises.
"""
import argparse
import os
import time

import numpy as np
import torch

import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.gan import Generator, Discriminator
from models.cyclegan import CycleGenerator, PatchDiscriminator
from models.ddpm import DiffusionSchedule, SimpleUNet
from models.classifier import train_classifier

def set_seed(seed=42):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ----------------------------------------------------------------------
#  GAN
# ----------------------------------------------------------------------
def train_gan(args, device):
    """Train the vanilla GAN on MNIST. Returns the path of the saved checkpoint."""
    Z_DIM = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = torchvision.datasets.MNIST(args.data_root, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f'MNIST: {len(dataset)} training images')

    G = Generator(Z_DIM).to(device)
    D = Discriminator().to(device)

    g_lr = args.g_lr if args.g_lr is not None else 2e-4
    d_lr = args.d_lr if args.d_lr is not None else 2e-4
    print(f'Generator LR = {g_lr} | Discriminator LR = {d_lr}')

    opt_G = torch.optim.Adam(G.parameters(), lr=g_lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    g_losses, d_losses, epoch_times = [], [], []

    for epoch in range(args.epochs):
        t0 = time.time()
        g_ep, d_ep = [], []
        for real_imgs, _ in loader:
            B = real_imgs.size(0)
            real_imgs = real_imgs.view(B, -1).to(device)
            real_labels = torch.ones(B, 1).to(device)
            fake_labels = torch.zeros(B, 1).to(device)

            # --- Train Discriminator ---
            z = torch.randn(B, Z_DIM).to(device)
            fake_imgs = G(z).detach()
            d_loss = criterion(D(real_imgs), real_labels) + criterion(D(fake_imgs), fake_labels)
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # --- Train Generator ---
            z = torch.randn(B, Z_DIM).to(device)
            g_loss = criterion(D(G(z)), real_labels)
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            g_ep.append(g_loss.item())
            d_ep.append(d_loss.item())

        ep_time = time.time() - t0
        epoch_times.append(ep_time)
        g_losses.append(float(np.mean(g_ep)))
        d_losses.append(float(np.mean(d_ep)))
        print(f'Epoch {epoch+1:02d}/{args.epochs} | G: {np.mean(g_ep):.3f} | '
              f'D: {np.mean(d_ep):.3f} | {ep_time:.1f}s')

    os.makedirs('saved', exist_ok=True)
    save_path = os.path.join('saved', args.save_name or 'gan_mnist.pt')
    torch.save({
        'G': G.state_dict(),
        'D': D.state_dict(),
        'z_dim': Z_DIM,
        'g_lr': g_lr,
        'd_lr': d_lr,
        'g_losses': g_losses,
        'd_losses': d_losses,
        'epoch_times': epoch_times,
    }, save_path)
    print(f'Saved GAN checkpoint to {save_path}')
    return save_path


def generate_gan_samples(args, device):
    """Load a trained GAN generator and produce n samples (saved as a grid image)."""
    ckpt = torch.load(args.weights, map_location=device)
    z_dim = ckpt.get('z_dim', 100)
    G = Generator(z_dim).to(device)
    G.load_state_dict(ckpt['G'])
    G.eval()

    with torch.no_grad():
        z = torch.randn(args.n, z_dim).to(device)
        imgs = G(z).view(-1, 1, 28, 28).cpu()

    import matplotlib.pyplot as plt
    grid = torchvision.utils.make_grid(imgs, nrow=8, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    os.makedirs('saved', exist_ok=True)
    out_path = os.path.join('saved', 'gan_generated_samples.png')
    plt.savefig(out_path, bbox_inches='tight')
    print(f'Saved generated sample grid to {out_path}')
    return imgs



# ----------------------------------------------------------------------
#  CycleGAN
# ----------------------------------------------------------------------
def get_celeba_loaders(args, img_size=64, n_per_domain=30000):
    """Build the dark-hair / blonde-hair CelebA loaders used by CycleGAN.

    Loads CelebA via the Hugging Face Hub (flwrlabs/celeba) instead of
    torchvision's built-in downloader, which depends on Google Drive and
    frequently fails with "quota exceeded" errors.
    """
    from datasets import load_dataset

    print('Loading CelebA from Hugging Face Hub (first run downloads it, can take a while)...')
    hf_ds = load_dataset('flwrlabs/celeba', split='train')

    celeba_transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    class HFCelebADataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, indices, transform):
            self.ds = hf_dataset
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            row = self.ds[self.indices[idx]]
            img = row['image'].convert('RGB')
            return self.transform(img), 0

    print('Scanning attributes for Blond_Hair split...')
    dark_indices, blonde_indices = [], []
    for i, row in enumerate(hf_ds):
        if row['Blond_Hair'] and len(blonde_indices) < n_per_domain:
            blonde_indices.append(i)
        elif not row['Blond_Hair'] and len(dark_indices) < n_per_domain:
            dark_indices.append(i)
        if len(dark_indices) >= n_per_domain and len(blonde_indices) >= n_per_domain:
            break

    celeba_dark = HFCelebADataset(hf_ds, dark_indices, celeba_transform)
    celeba_blonde = HFCelebADataset(hf_ds, blonde_indices, celeba_transform)

    loader_dark = DataLoader(celeba_dark, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, drop_last=True)
    loader_blonde = DataLoader(celeba_blonde, batch_size=args.batch_size, shuffle=True,
                                num_workers=2, drop_last=True)
    print(f'Domain X (dark hair):   {len(celeba_dark)} images')
    print(f'Domain Y (blonde hair): {len(celeba_blonde)} images')
    return loader_dark, loader_blonde, celeba_dark, celeba_blonde


def train_cyclegan(args, device):
    """Train CycleGAN on CelebA dark-hair <-> blonde-hair translation.

    args.lambda_cyc controls the cycle-consistency weight -- set to 0 for
    Exercise 2's ablation (ordinarily 10.0, the paper's default).
    """
    loader_dark, loader_blonde, _, _ = get_celeba_loaders(args)

    G = CycleGenerator().to(device)   # dark -> blonde
    Fn = CycleGenerator().to(device)  # blonde -> dark
    D_X = PatchDiscriminator().to(device)
    D_Y = PatchDiscriminator().to(device)

    lambda_cyc = args.lambda_cyc if args.lambda_cyc is not None else 10.0
    lambda_idt = args.lambda_idt if args.lambda_idt is not None else 5.0
    print(f'LAMBDA_CYC = {lambda_cyc} | LAMBDA_IDT = {lambda_idt}')

    opt_G_all = torch.optim.Adam(list(G.parameters()) + list(Fn.parameters()), lr=2e-4, betas=(0.5, 0.999))
    opt_D_all = torch.optim.Adam(list(D_X.parameters()) + list(D_Y.parameters()), lr=2e-4, betas=(0.5, 0.999))

    adv_loss = nn.MSELoss()   # LSGAN-style: smoother gradients than BCE
    cyc_loss = nn.L1Loss()

    g_losses, d_losses, epoch_times = [], [], []

    for epoch in range(args.epochs):
        t0 = time.time()
        g_ep, d_ep = [], []

        dark_iter = iter(loader_dark)
        blonde_iter = iter(loader_blonde)
        n_batches = min(len(loader_dark), len(loader_blonde))

        for _ in range(n_batches):
            real_x, _ = next(dark_iter)
            real_y, _ = next(blonde_iter)
            real_x, real_y = real_x.to(device), real_y.to(device)

            # ---- Train Generators G and F ----
            opt_G_all.zero_grad()

            fake_y = G(real_x)          # dark -> blonde
            fake_x = Fn(real_y)         # blonde -> dark
            cycle_x = Fn(fake_y)        # dark -> blonde -> dark
            cycle_y = G(fake_x)         # blonde -> dark -> blonde
            idt_x = Fn(real_x)
            idt_y = G(real_y)

            patch_shape = D_Y(fake_y).shape
            real_label = torch.ones(patch_shape, device=device)
            fake_label = torch.zeros(patch_shape, device=device)

            loss_G_adv = adv_loss(D_Y(fake_y), real_label) + adv_loss(D_X(fake_x), real_label)
            loss_cyc = cyc_loss(cycle_x, real_x) + cyc_loss(cycle_y, real_y)
            loss_idt = cyc_loss(idt_x, real_x) + cyc_loss(idt_y, real_y)
            loss_G = loss_G_adv + lambda_cyc * loss_cyc + lambda_idt * loss_idt

            loss_G.backward()
            opt_G_all.step()

            # ---- Train Discriminators D_X and D_Y ----
            opt_D_all.zero_grad()
            loss_D_X = adv_loss(D_X(real_x), real_label) + adv_loss(D_X(fake_x.detach()), fake_label)
            loss_D_Y = adv_loss(D_Y(real_y), real_label) + adv_loss(D_Y(fake_y.detach()), fake_label)
            loss_D = (loss_D_X + loss_D_Y) * 0.5
            loss_D.backward()
            opt_D_all.step()

            g_ep.append(loss_G.item())
            d_ep.append(loss_D.item())

        ep_time = time.time() - t0
        epoch_times.append(ep_time)
        g_losses.append(float(np.mean(g_ep)))
        d_losses.append(float(np.mean(d_ep)))
        print(f'Epoch {epoch+1:02d}/{args.epochs} | G: {np.mean(g_ep):.3f} | '
              f'D: {np.mean(d_ep):.3f} | {ep_time:.1f}s')

    os.makedirs('saved', exist_ok=True)
    save_path = os.path.join('saved', args.save_name or 'cyclegan_celeba.pt')
    torch.save({
        'G': G.state_dict(), 'F': Fn.state_dict(),
        'D_X': D_X.state_dict(), 'D_Y': D_Y.state_dict(),
        'lambda_cyc': lambda_cyc, 'lambda_idt': lambda_idt,
        'g_losses': g_losses, 'd_losses': d_losses, 'epoch_times': epoch_times,
    }, save_path)
    print(f'Saved CycleGAN checkpoint to {save_path}')
    return save_path


def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)


def test_cyclegan_image(args, device):
    """Run a single face photo through both trained generators (G: ->blonde, F: ->dark)."""
    from PIL import Image
    import matplotlib.pyplot as plt

    ckpt = torch.load(args.weights, map_location=device)
    G = CycleGenerator().to(device)
    Fn = CycleGenerator().to(device)
    G.load_state_dict(ckpt['G'])
    Fn.load_state_dict(ckpt['F'])
    G.eval(); Fn.eval()

    img_size = 64
    face_transform = transforms.Compose([
        transforms.CenterCrop(min(Image.open(args.test_image).size)),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img_pil = Image.open(args.test_image).convert('RGB')
    img_tensor = face_transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        to_blonde = G(img_tensor).squeeze(0).cpu()
        to_dark = Fn(img_tensor).squeeze(0).cpu()

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    titles = ['Original', 'G: -> Blonde Hair', 'F: -> Dark Hair']
    imgs = [img_tensor.squeeze(0).cpu(), to_blonde, to_dark]
    for ax, title, im in zip(axes, titles, imgs):
        ax.imshow(denorm(im).permute(1, 2, 0))
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    plt.tight_layout()
    os.makedirs('saved', exist_ok=True)
    out_path = os.path.join('saved', 'cyclegan_my_face_result.png')
    plt.savefig(out_path, bbox_inches='tight')
    print(f'Saved your face translation result to {out_path}')


# ----------------------------------------------------------------------
#  DDPM
# ----------------------------------------------------------------------
def train_ddpm(args, device, timesteps=1000):
    """Train a DDPM on MNIST. args.schedule picks 'linear' or 'cosine' noise schedule."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = torchvision.datasets.MNIST(args.data_root, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f'MNIST: {len(dataset)} training images')

    schedule = DiffusionSchedule(timesteps=timesteps, schedule=args.schedule, device=device)
    model = SimpleUNet(in_ch=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    mse_loss = nn.MSELoss()

    print(f'Noise schedule: {args.schedule} | timesteps: {timesteps}')

    losses, epoch_times = [], []
    for epoch in range(args.epochs):
        t0 = time.time()
        ep_losses = []
        for imgs, _ in loader:
            imgs = imgs.to(device)
            B = imgs.size(0)

            # Pick a random noise level per image, add that much noise, then
            # ask the model to predict exactly what noise was added.
            t = torch.randint(0, timesteps, (B,), device=device).long()
            noisy_imgs, noise = schedule.q_sample(imgs, t)
            pred_noise = model(noisy_imgs, t)
            loss = mse_loss(pred_noise, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_losses.append(loss.item())

        ep_time = time.time() - t0
        epoch_times.append(ep_time)
        losses.append(float(np.mean(ep_losses)))
        print(f'Epoch {epoch+1:02d}/{args.epochs} | Loss: {np.mean(ep_losses):.4f} | {ep_time:.1f}s')

    os.makedirs('saved', exist_ok=True)
    save_path = os.path.join('saved', args.save_name or f'ddpm_mnist_{args.schedule}.pt')
    torch.save({
        'model': model.state_dict(),
        'schedule': args.schedule,
        'timesteps': timesteps,
        'losses': losses,
        'epoch_times': epoch_times,
    }, save_path)
    print(f'Saved DDPM checkpoint to {save_path}')
    return save_path



def generate_ddpm_samples(args, device, timesteps=1000):
    """Load a trained DDPM and generate n images by reversing the noise process
    step by step, starting from pure random noise."""
    ckpt = torch.load(args.weights, map_location=device)
    schedule_name = ckpt.get('schedule', 'linear')
    timesteps = ckpt.get('timesteps', timesteps)

    schedule = DiffusionSchedule(timesteps=timesteps, schedule=schedule_name, device=device)
    model = SimpleUNet(in_ch=1).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    n = args.n
    x = torch.randn(n, 1, 28, 28, device=device)

    betas = schedule.betas
    alphas = schedule.alphas
    alphas_cumprod = schedule.alphas_cumprod
    # alpha_bar at t-1, defined as 1 before t=0 (no noise yet)
    alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]])

    with torch.no_grad():
        for t_step in reversed(range(timesteps)):
            t_batch = torch.full((n,), t_step, device=device, dtype=torch.long)
            pred_noise = model(x, t_batch)

            alpha_bar_t = alphas_cumprod[t_step]
            alpha_bar_prev = alphas_cumprod_prev[t_step]
            alpha_t = alphas[t_step]
            beta_t = betas[t_step]

            # Reconstruct the model's best guess at the clean image, then clip it.
            # This clip is the key stability fix -- it stops a bad prediction at a
            # tiny alpha_t step from exploding the rest of the reverse process.
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            x0_pred = x0_pred.clamp(-1, 1)

            # Posterior mean: a numerically safer blend of x0_pred and the
            # current noisy image (standard DDPM posterior formula).
            coef_x0 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1 - alpha_bar_t)
            coef_xt = (torch.sqrt(alpha_t) * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
            mean = coef_x0 * x0_pred + coef_xt * x

            if t_step > 0:
                posterior_var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(posterior_var) * noise
            else:
                x = mean

    imgs = x.clamp(-1, 1).cpu()

    import matplotlib.pyplot as plt
    grid = torchvision.utils.make_grid(imgs, nrow=8, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0), cmap='gray')
    plt.axis('off')
    os.makedirs('saved', exist_ok=True)
    out_path = os.path.join('saved', f'ddpm_generated_{schedule_name}.png')
    plt.savefig(out_path, bbox_inches='tight')
    print(f'Saved generated DDPM samples to {out_path}')
    return imgs

def main():
    parser = argparse.ArgumentParser(description='A4 Generative Models runner')
    parser.add_argument('--model', choices=['gan', 'cyclegan', 'ddpm'], help='Which model to run')
    parser.add_argument('--dataset', choices=['mnist', 'celeba'], default='mnist')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--train', action='store_true', help='Train the selected model')
    parser.add_argument('--generate', action='store_true', help='Generate samples from a trained model')
    parser.add_argument('--n', type=int, default=64, help='Number of samples to generate')
    parser.add_argument('--weights', type=str, default=None, help='Path to a saved checkpoint')
    parser.add_argument('--save-name', type=str, default=None, help='Filename to save checkpoint under saved/')
    parser.add_argument('--test-image', type=str, default=None, help='Path to a face photo (CycleGAN)')
    parser.add_argument('--schedule', choices=['linear', 'cosine'], default='linear', help='DDPM noise schedule')
    parser.add_argument('--data-root', type=str, default='./data', dest='data_root')
    parser.add_argument('--seed', type=int, default=42)

    # GAN-specific (Exercise 1b uses --d_lr to induce mode collapse)
    parser.add_argument('--g_lr', type=float, default=None, help='Generator learning rate (GAN)')
    parser.add_argument('--d_lr', type=float, default=None, help='Discriminator learning rate (GAN)')


    # CycleGAN-specific (Exercise 2 uses --lambda_cyc=0 for the ablation)
    parser.add_argument('--lambda_cyc', type=float, default=None, help='Cycle consistency loss weight (CycleGAN)')
    parser.add_argument('--lambda_idt', type=float, default=None, help='Identity loss weight (CycleGAN)')

    # Helper classifier (used by Exercise 1's mode-collapse analysis)
    parser.add_argument('--train-classifier', action='store_true',
                         help='Train the small MNIST CNN used to judge generated digits')

    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f'Using device: {device}')

    if args.train_classifier:
        train_classifier(device, data_root=args.data_root, epochs=args.epochs or 3)
        return

    if args.model == 'gan':
        if args.train:
            train_gan(args, device)
        elif args.generate:
            assert args.weights, '--weights is required with --generate'
            generate_gan_samples(args, device)
        else:
            parser.error('Specify --train or --generate for --model gan')

    elif args.model == 'cyclegan':
        if args.train:
            train_cyclegan(args, device)
        elif args.test_image:
            assert args.weights, '--weights is required with --test-image'
            test_cyclegan_image(args, device)
        else:
            parser.error('Specify --train or --test-image for --model cyclegan')

    elif args.model == 'ddpm':
        if args.train:
            train_ddpm(args, device)
        elif args.generate:
            assert args.weights, '--weights is required with --generate'
            generate_ddpm_samples(args, device)
        else:
            parser.error('Specify --train or --generate for --model ddpm')

    else:
        parser.error('--model is required (choose: gan, cyclegan, ddpm)')


if __name__ == '__main__':
    main()