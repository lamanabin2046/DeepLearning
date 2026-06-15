from pathlib import Path

p = Path("run.py")
s = p.read_text()

mae_code = r'''
# -----------------------------
# MAE Dataset
# -----------------------------
class CIFAR10MAE(CIFAR10):
    def __init__(self, root="./data", train=True, download=True):
        transform = T.Compose([
            T.RandomResizedCrop(32, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2470, 0.2435, 0.2616))
        ])
        super().__init__(root=root, train=train, download=download, transform=transform)


# -----------------------------
# Masked Autoencoder
# -----------------------------
class MAEModel(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3,
                 embed_dim=192, depth=4, num_heads=3,
                 decoder_dim=192, decoder_depth=2, decoder_heads=3,
                 mask_ratio=0.75):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_chans

        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

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
        self.encoder_norm = nn.LayerNorm(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))

        dec_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def patchify(self, imgs):
        # imgs: [B, 3, 32, 32]
        p = self.patch_size
        B, C, H, W = imgs.shape
        h = H // p
        w = W // p

        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, h * w, p * p * C)
        return x

    def unpatchify(self, patches):
        # patches: [B, N, patch_dim]
        p = self.patch_size
        B, N, D = patches.shape
        h = w = int(N ** 0.5)
        C = self.in_chans

        x = patches.reshape(B, h, w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, C, h * p, w * p)
        return x

    def random_masking(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_visible, mask, ids_restore

    def forward(self, imgs):
        target = self.patchify(imgs)                       # [B, 64, 48]
        x = self.patch_embed(target) + self.pos_embed      # [B, 64, 192]

        x_visible, mask, ids_restore = self.random_masking(x)

        encoded = self.encoder(x_visible)
        encoded = self.encoder_norm(encoded)

        x_dec = self.decoder_embed(encoded)

        B, L, D = x_dec.shape
        N = self.num_patches

        mask_tokens = self.mask_token.repeat(B, N - L, 1)
        x_full = torch.cat([x_dec, mask_tokens], dim=1)

        x_full = torch.gather(
            x_full,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, D)
        )

        x_full = x_full + self.decoder_pos_embed
        x_full = self.decoder(x_full)
        x_full = self.decoder_norm(x_full)

        pred = self.decoder_pred(x_full)                   # [B, 64, 48]

        loss_per_patch = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss_per_patch * mask).sum() / mask.sum()

        return loss, pred, mask, target

    def features(self, imgs):
        # For linear evaluation: use full image, no masking
        patches = self.patchify(imgs)
        x = self.patch_embed(patches) + self.pos_embed
        x = self.encoder(x)
        x = self.encoder_norm(x)
        feat = x.mean(dim=1)
        return feat


def make_mae_tag(mask_ratio):
    return f"mae_encoder_{int(round(mask_ratio * 100)):03d}"


def train_mae(args):
    make_dirs()
    set_seed(args.seed)
    device = get_device()

    tag = make_mae_tag(args.mask_ratio)

    print("=" * 60)
    print("Training MAE")
    print("Setting:", tag)
    print("Mask ratio:", args.mask_ratio)
    print("Device:", device)
    print("Epochs:", args.epochs)
    print("=" * 60)

    dataset = CIFAR10MAE(root="./data", train=True, download=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    model = MAEModel(mask_ratio=args.mask_ratio).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    losses = []
    epoch_times = []

    for epoch in range(args.epochs):
        model.train()
        ep_losses = []
        start = time.time()

        for images, _ in loader:
            images = images.to(device, non_blocking=True)

            loss, pred, mask, target = model(images)

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
            f"Reconstruction Loss: {avg_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

    save_path = f"saved/{tag}.pt"

    torch.save({
        "model": model.state_dict(),
        "losses": losses,
        "epoch_times": epoch_times,
        "mask_ratio": args.mask_ratio,
        "args": vars(args)
    }, save_path)

    print("Saved model:", save_path)

    plot_curve(
        losses,
        title=f"{tag} Reconstruction Loss",
        ylabel="Reconstruction Loss",
        save_path=f"results/{tag}_loss.png"
    )

    save_mae_reconstruction(
        model=model,
        device=device,
        save_path=f"results/{tag}_reconstruction.png"
    )

    print("Saved plots:")
    print(f"results/{tag}_loss.png")
    print(f"results/{tag}_reconstruction.png")


@torch.no_grad()
def save_mae_reconstruction(model, device, save_path):
    model.eval()

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616))
    ])

    dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    images, _ = next(iter(loader))
    images = images.to(device)

    loss, pred, mask, target = model(images)

    mask_3d = mask.unsqueeze(-1)

    masked_patches = target * (1 - mask_3d)
    recon_patches = target * (1 - mask_3d) + pred * mask_3d

    masked_imgs = model.unpatchify(masked_patches)
    recon_imgs = model.unpatchify(recon_patches)

    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)

    original = (images * std + mean).clamp(0, 1)
    masked = (masked_imgs * std + mean).clamp(0, 1)
    recon = (recon_imgs * std + mean).clamp(0, 1)

    rows = [original, masked, recon]
    titles = ["Original", "Masked", "Reconstructed"]

    plt.figure(figsize=(12, 5))

    for r in range(3):
        for c in range(8):
            ax = plt.subplot(3, 8, r * 8 + c + 1)
            img = rows[r][c].detach().cpu().permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(titles[r], fontsize=10)

    plt.suptitle("MAE Reconstruction: Original → Masked → Reconstructed")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def linear_eval_mae(args):
    make_dirs()
    set_seed(args.seed)
    device = get_device()

    if args.weights is None:
        raise ValueError("Please provide --weights saved/mae_encoder_075.pt")

    print("=" * 60)
    print("MAE Linear Evaluation")
    print("Weights:", args.weights)
    print("Device:", device)
    print("=" * 60)

    model = MAEModel(mask_ratio=args.mask_ratio).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model"])
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
        batch_losses = []

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

            batch_losses.append(loss.item())
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
            f"Loss: {np.mean(batch_losses):.4f} | "
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
'''

if "class MAEModel" not in s:
    marker = "# -----------------------------\n# Main\n# -----------------------------"
    if marker not in s:
        raise RuntimeError("Could not find Main marker in run.py")
    s = s.replace(marker, mae_code + "\n\n" + marker)

s = s.replace('choices=["dino"]', 'choices=["dino", "mae"]')

if '--mask-ratio' not in s:
    s = s.replace(
        'parser.add_argument("--no-centering", action="store_true")',
        'parser.add_argument("--no-centering", action="store_true")\n    parser.add_argument("--mask-ratio", type=float, default=0.75)'
    )

old_dispatch = '''    if args.model == "dino" and args.train:
        train_dino(args)

    elif args.model == "dino" and args.evaluate and args.linear:
        linear_eval(args)

    else:
        print("Please choose one mode:")
        print("Train DINO:")
        print("  python run.py --model dino --epochs 50 --train")
        print("Linear eval:")
        print("  python run.py --model dino --weights saved/dino_default.pt --evaluate --linear")'''

new_dispatch = '''    if args.model == "dino" and args.train:
        train_dino(args)

    elif args.model == "dino" and args.evaluate and args.linear:
        linear_eval(args)

    elif args.model == "mae" and args.train:
        train_mae(args)

    elif args.model == "mae" and args.evaluate and args.linear:
        linear_eval_mae(args)

    else:
        print("Please choose one mode:")
        print("Train DINO:")
        print("  python run.py --model dino --epochs 50 --train")
        print("Linear eval DINO:")
        print("  python run.py --model dino --weights saved/dino_default.pt --evaluate --linear")
        print("Train MAE:")
        print("  python run.py --model mae --mask-ratio 0.75 --epochs 50 --train")
        print("Linear eval MAE:")
        print("  python run.py --model mae --weights saved/mae_encoder_075.pt --evaluate --linear")'''

if old_dispatch in s:
    s = s.replace(old_dispatch, new_dispatch)
else:
    print("Warning: dispatch block not found exactly. Please check run.py manually.")

p.write_text(s)
print("MAE code added to run.py successfully.")
