"""
Exercise 1 -- GAN Mode Collapse

a) Train the vanilla GAN normally, generate 1000 images, classify each with
   a pretrained MNIST CNN, and report/plot the digit-count histogram.
b) Intentionally trigger mode collapse with a 3x discriminator learning
   rate (6e-4 instead of 2e-4), retrain, and report the histogram again.
c) (answered in the README / written discussion, not code)

Usage (run from the project root, AFTER you have trained both GAN
checkpoints and the classifier -- see the printed commands below):

    # 1. one-time: train the judge classifier
    python run.py --train-classifier --epochs 3

    # 2. train the normal GAN (part a)
    python run.py --model gan --dataset mnist --epochs 20 --train \
                   --save-name gan_mnist_normal.pt

    # 3. train the collapsed GAN (part b): 3x discriminator LR
    python run.py --model gan --dataset mnist --epochs 20 --train \
                   --d_lr 6e-4 --save-name gan_mnist_collapsed.pt

    # 4. run this script to produce both histograms + the comparison table
    python exercises/exercise1_mode_collapse.py
"""
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gan import Generator
from models.classifier import load_classifier


def classify_generator_outputs(generator, classifier, device, n_samples=1000, batch_size=200):
    """Generate n_samples images from `generator` and label each with `classifier`.

    Returns a length-10 numpy array: counts[d] = how many of the n_samples
    images were classified as digit d.
    """
    counts = np.zeros(10, dtype=int)
    generator.eval()
    classifier.eval()
    remaining = n_samples
    with torch.no_grad():
        while remaining > 0:
            b = min(batch_size, remaining)
            z = torch.randn(b, generator.z_dim, device=device)
            imgs = generator(z).view(-1, 1, 28, 28)  # already in [-1, 1], matches classifier training norm
            preds = classifier(imgs).argmax(dim=1).cpu().numpy()
            for p in preds:
                counts[p] += 1
            remaining -= b
    return counts


def plot_histogram(counts, title, save_path):
    plt.figure(figsize=(8, 4))
    plt.bar(range(10), counts, color='steelblue')
    plt.xticks(range(10))
    plt.xlabel('Digit')
    plt.ylabel('Count (out of 1000)')
    plt.title(title)
    plt.grid(axis='y', alpha=0.3)
    for i, c in enumerate(counts):
        plt.text(i, c + 5, str(c), ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f'Saved histogram to {save_path}')


def print_table(counts, label):
    print(f'\n--- {label} ---')
    header = '| Digit | ' + ' | '.join(str(d) for d in range(10)) + ' |'
    sep = '|---' * 11 + '|'
    row = '| Count | ' + ' | '.join(str(c) for c in counts) + ' |'
    print(header)
    print(sep)
    print(row)
    vanished = [d for d in range(10) if counts[d] == 0]
    rare = [d for d in range(10) if 0 < counts[d] < 10]
    print(f'Digits with ZERO samples (fully vanished): {vanished if vanished else "none"}')
    print(f'Digits with <10 samples (nearly vanished): {rare if rare else "none"}')
    print(f'Coverage spread (max-min count): {counts.max() - counts.min()}')
    print(f'Std-dev across digits: {counts.std():.1f}  (0 = perfectly even coverage)')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('saved', exist_ok=True)

    classifier_path = 'saved/mnist_classifier.pt'
    if not os.path.exists(classifier_path):
        print(f'ERROR: {classifier_path} not found.')
        print('Run first:  python run.py --train-classifier --epochs 3')
        return
    classifier = load_classifier(device, classifier_path)

    results = {}
    configs = [
        ('saved/gan_mnist_normal.pt', 'Normal GAN (d_lr=2e-4)', 'saved/exercise1_normal_hist.png'),
        ('saved/gan_mnist_collapsed.pt', 'Collapsed GAN (d_lr=6e-4)', 'saved/exercise1_collapsed_hist.png'),
    ]

    for ckpt_path, label, hist_path in configs:
        if not os.path.exists(ckpt_path):
            print(f'\nSkipping "{label}": checkpoint not found at {ckpt_path}')
            print('Train it first with the commands in this script\'s docstring.')
            continue
        ckpt = torch.load(ckpt_path, map_location=device)
        G = Generator(ckpt.get('z_dim', 100)).to(device)
        G.load_state_dict(ckpt['G'])

        counts = classify_generator_outputs(G, classifier, device, n_samples=1000)
        results[label] = counts
        print_table(counts, label)
        plot_histogram(counts, f'Digit distribution -- {label}', hist_path)

    if len(results) == 2:
        labels = list(results.keys())
        fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
        for ax, label in zip(axes, labels):
            ax.bar(range(10), results[label], color='steelblue' if 'Normal' in label else 'coral')
            ax.set_xticks(range(10))
            ax.set_xlabel('Digit')
            ax.set_title(label)
            ax.grid(axis='y', alpha=0.3)
        axes[0].set_ylabel('Count (out of 1000)')
        plt.suptitle('Exercise 1 -- Mode Collapse Comparison')
        plt.tight_layout()
        out_path = 'saved/exercise1_comparison.png'
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f'\nSaved side-by-side comparison plot to {out_path}')


if __name__ == '__main__':
    main()