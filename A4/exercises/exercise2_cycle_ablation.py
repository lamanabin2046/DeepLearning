"""
Exercise 2 -- CycleGAN Ablation: Cycle Consistency

a) Compare translation quality between the default setting (LAMBDA_CYC=10)
   and the ablated setting (LAMBDA_CYC=0).
b) Show 4 example translations from each setting, both directions
   (dark->blonde and blonde->dark).
c) (answered in the README / written discussion, not code)

Usage (run from the project root, AFTER training BOTH checkpoints):

    # 1. default setting (paper's recommended weight)
    python run.py --model cyclegan --dataset celeba --epochs 10 --train \
                   --lambda_cyc 10 --save-name cyclegan_default.pt

    # 2. ablation: cycle consistency disabled
    python run.py --model cyclegan --dataset celeba --epochs 10 --train \
                   --lambda_cyc 0 --save-name cyclegan_no_cycle.pt

    # 3. run this script to build the comparison grid
    python exercises/exercise2_cycle_ablation.py
"""
import os
import sys

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cyclegan import CycleGenerator
from run import get_celeba_loaders, denorm


def load_generators(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    G = CycleGenerator().to(device)
    F = CycleGenerator().to(device)
    G.load_state_dict(ckpt['G'])
    F.load_state_dict(ckpt['F'])
    G.eval(); F.eval()
    return G, F, ckpt.get('lambda_cyc', None)


def build_translation_grid(G, F, celeba_dark, celeba_blonde, device, n=4):
    """4 rows: real dark | dark->blonde | real blonde | blonde->dark, n columns."""
    with torch.no_grad():
        batch_x, _ = next(iter(DataLoader(celeba_dark, batch_size=n, shuffle=True)))
        batch_y, _ = next(iter(DataLoader(celeba_blonde, batch_size=n, shuffle=True)))
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        fake_y = G(batch_x).cpu()
        fake_x = F(batch_y).cpu()
    return batch_x.cpu(), fake_y, batch_y.cpu(), fake_x


def plot_grid(imgs_list, row_labels, title, save_path):
    n_rows, n_cols = len(imgs_list), imgs_list[0].size(0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))
    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            ax.imshow(denorm(imgs_list[row][col]).permute(1, 2, 0))
            ax.axis('off')
        axes[row, 0].set_ylabel(row_labels[row], fontsize=9)
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f'Saved {save_path}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configs = [
        ('saved/cyclegan_default.pt', 'lambda_cyc = 10 (default)', 'saved/exercise2_default_grid.png'),
        ('saved/cyclegan_no_cycle.pt', 'lambda_cyc = 0 (ablated)', 'saved/exercise2_ablated_grid.png'),
    ]

    # Build the CelebA loaders once (reused for both checkpoints so the
    # comparison uses different random samples, which is fine for a
    # qualitative visual check)
    import argparse
    args = argparse.Namespace(data_root='./data', batch_size=16)

    for ckpt_path, label, out_path in configs:
        if not os.path.exists(ckpt_path):
            print(f'Skipping "{label}": checkpoint not found at {ckpt_path}')
            print('Train it first with the commands in this script\'s docstring.')
            continue

        print(f'\nLoading {label} ...')
        G, F, lambda_cyc_stored = load_generators(ckpt_path, device)
        print(f'(checkpoint reports lambda_cyc={lambda_cyc_stored})')

        _, _, celeba_dark, celeba_blonde = get_celeba_loaders(args)
        imgs = build_translation_grid(G, F, celeba_dark, celeba_blonde, device, n=4)
        row_labels = ['Real dark', 'Dark -> Blonde', 'Real blonde', 'Blonde -> Dark']
        plot_grid(imgs, row_labels, f'CycleGAN translations -- {label}', out_path)

    print('\nDone. Compare saved/exercise2_default_grid.png vs saved/exercise2_ablated_grid.png')
    print('Look for: face structure preserved? color bleeding? background distortion?')


if __name__ == '__main__':
    main()