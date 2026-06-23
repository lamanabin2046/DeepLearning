"""
DDPM (Denoising Diffusion Probabilistic Model) for MNIST.
Exercise 4 compares two noise schedules: linear vs cosine.
"""
import math
import torch
import torch.nn as nn


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """The original DDPM paper's schedule: beta grows linearly over time."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """Improved DDPM (Nichol & Dhariwal) cosine schedule.

    Instead of beta increasing linearly, this shapes the cumulative noise
    (alpha_bar) as a cosine curve, which adds noise more gently early on
    and more aggressively near the end -- intended to avoid destroying
    image information too quickly in early timesteps.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionSchedule:
    """Precomputes every quantity needed for forward noising and reverse sampling."""
    def __init__(self, timesteps=1000, schedule='linear', device='cpu'):
        self.timesteps = timesteps
        if schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps)

        self.betas = betas.to(device)
        self.alphas = (1.0 - betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)

    def q_sample(self, x0, t, noise=None):
        """Forward process: add noise to x0 at timestep t (closed-form, no loop needed)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_omac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ac * x0 + sqrt_omac * noise, noise



class SinusoidalTimeEmbedding(nn.Module):
    """Converts a scalar timestep t into a vector, the same way Transformers
    embed token positions -- so the network can tell 'how noisy is this image'."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class Block(nn.Module):
    """One conv block, with the timestep embedding injected partway through."""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]  # inject "how noisy" info here
        h = self.act(self.norm2(self.conv2(h)))
        return h


class SimpleUNet(nn.Module):
    """Small UNet: predicts the noise that was added to an image at timestep t."""
    def __init__(self, in_ch=1, base_ch=64, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.down1 = Block(in_ch, base_ch, time_dim)
        self.down2 = Block(base_ch, base_ch * 2, time_dim)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = Block(base_ch * 2, base_ch * 2, time_dim)
        self.up_conv = nn.Upsample(scale_factor=2, mode='nearest')
        self.up1 = Block(base_ch * 2 + base_ch * 2, base_ch * 2, time_dim)
        self.up2 = Block(base_ch * 2 + base_ch, base_ch, time_dim)
        self.out = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        d1 = self.down1(x, t_emb)                          # 28x28
        d2 = self.down2(self.pool(d1), t_emb)               # 14x14
        b = self.bottleneck(self.pool(d2), t_emb)           # 7x7
        u1 = self.up1(torch.cat([self.up_conv(b), d2], dim=1), t_emb)   # 14x14
        u2 = self.up2(torch.cat([self.up_conv(u1), d1], dim=1), t_emb)  # 28x28
        return self.out(u2)