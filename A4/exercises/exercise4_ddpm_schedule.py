"""
Exercise 4 -- DDPM Noise Schedule Ablation: Linear vs Cosine

Compares two noise schedules for diffusion training:
a) Plot both alpha_bar_t curves to visualize how each schedule destroys
   signal over time.
b) Compare training loss curves between the two schedules.
c) Generate samples from both trained models and compare visual quality.

Usage (run from the project root, AFTER training BOTH checkpoints):

    python run.py --model ddpm --dataset mnist --epochs 20 --schedule linear --train \
                   --save-name ddpm_mnist_linear.pt

    python run.py --model ddpm --dataset mnist --epochs 20 --schedule cosine --train \
                   --save-name ddpm_mnist_cosine.pt

    python exercises/exercise4_ddpm_schedule.py
"""
import os
import sys

import torch
import torchvision
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ddpm import DiffusionSchedule
from run import generate_ddpm_samples
import argparse


def plot_alpha_bar_curves(timesteps=1000, save_path='saved/exercise4_schedules_comparison.png'):
    linear_sched = DiffusionSchedule(timesteps, 'linear')
    cosine_sched = DiffusionSchedule(timesteps, 'cosine')

    plt.figure(figsize=(7, 4))
    plt.plot(linear_sched.alphas_cumprod.numpy(), label='Linear')
    plt.plot(cosine_sched.alphas_cumprod.numpy(), label='Cosine')
    plt.xlabel('Timestep t')
    plt.ylabel('alpha_bar_t  (fraction of original signal retained)')
    plt.title('Noise schedule comparison: how fast signal is destroyed')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f'Saved {save_path}')


def plot_loss_curves(ckpt_paths, labels, save_path='saved/exercise4_loss_comparison.png'):
    plt.figure(figsize=(7, 4))
    any_plotted = False
    for ckpt_path, label in zip(ckpt_paths, labels):
        if not os.path.exists(ckpt_path):
            print(f'Skipping "{label}": {ckpt_path} not found')
            continue
        ckpt = torch.load(ckpt_path, map_location='cpu')
        plt.plot(ckpt['losses'], label=label)
        any_plotted = True
    if not any_plotted:
        print('No checkpoints found yet -- train both schedules first.')
        return
    plt.xlabel('Epoch')
    plt.ylabel('MSE loss (noise prediction)')
    plt.title('Training loss: linear vs cosine schedule')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f'Saved {save_path}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('saved', exist_ok=True)

    # (a) Schedule shape comparison -- doesn't need any trained model
    plot_alpha_bar_curves()

    ckpts = [
        ('saved/ddpm_mnist_linear.pt', 'Linear'),
        ('saved/ddpm_mnist_cosine.pt', 'Cosine'),
    ]

    # (b) Training loss comparison
    plot_loss_curves([c[0] for c in ckpts], [c[1] for c in ckpts])

    # (c) Generated sample comparison
    imgs_dict = {}
    for ckpt_path, label in ckpts:
        if not os.path.exists(ckpt_path):
            print(f'Skipping "{label}": checkpoint not found at {ckpt_path}')
            continue
        gen_args = argparse.Namespace(weights=ckpt_path, n=16)
        imgs_dict[label] = generate_ddpm_samples(gen_args, device)

    if len(imgs_dict) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, (label, imgs) in zip(axes, imgs_dict.items()):
            grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True)
            ax.imshow(grid.permute(1, 2, 0), cmap='gray')
            ax.set_title(label)
            ax.axis('off')
        plt.suptitle('DDPM generated digits: linear vs cosine schedule')
        plt.tight_layout()
        out_path = 'saved/exercise4_samples_comparison.png'
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f'Saved {out_path}')


if __name__ == '__main__':
    main()