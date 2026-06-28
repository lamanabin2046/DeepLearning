#!/usr/bin/env python3
"""
run.py — single CLI entry point for A6 Speech Processing.

Usage examples (exactly as listed in the assignment's Submission section):

    python3 run.py --model ctc --epochs 300 --train

    python3 run.py --model wav2vec2-probe --dataset speechcommands --classes yes,no,stop,go --train

    python3 run.py --model voice-clone --extract-se --reference my_voice.wav

    python3 run.py --model voice-clone --accent us --text "I got the job!" --generate

    python3 run.py --model voice-clone --accent all --text "Hello world" --generate

    python3 run.py --model voice-clone --language es --text "Hola, como estas?" --generate

Each `--model` value dispatches to its own self-contained function below. Only the
imports actually needed for the chosen mode are loaded, so e.g. running the `ctc`
mode does not require torchaudio, transformers, or OpenVoice to be installed.
"""

import argparse
import json
import os
import random
import sys

import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Shared utilities (Part 3 / CTC)
# ──────────────────────────────────────────────────────────────────────────

BLANK = '_'


def ctc_collapse(alignment):
    """Merge consecutive duplicates, then remove blanks. alignment: list of chars."""
    merged = []
    for ch in alignment:
        if not merged or ch != merged[-1]:
            merged.append(ch)
    return ''.join(ch for ch in merged if ch != BLANK)


def edit_distance(a, b):
    """Levenshtein distance between two strings (used for character error rate)."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


# ──────────────────────────────────────────────────────────────────────────
# Mode 1: --model ctc   (Part 3 / Exercise 2)
# ──────────────────────────────────────────────────────────────────────────

def run_ctc(args):
    """Train the toy frame-to-character CTC model and report CER over training."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ALPHABET = list('helo wrd')
    CHAR2IDX = {c: i + 1 for i, c in enumerate(ALPHABET)}  # 0 reserved for blank
    IDX2CHAR = {i + 1: c for i, c in enumerate(ALPHABET)}
    VOCAB_SIZE = len(ALPHABET) + 1
    N_MELS = 20
    WORDS = ['hello', 'world', 'hero', 'red', 'led', 'doer']

    def synthesize_frames(word, frames_per_char=(args.min_frames, args.max_frames)):
        frames = []
        for ch in word:
            n = random.randint(*frames_per_char)
            base = np.zeros(N_MELS)
            base[CHAR2IDX[ch] % N_MELS] = 3.0
            for _ in range(n):
                frames.append(base + np.random.randn(N_MELS) * 0.5)
        return np.stack(frames)

    class TinyCTCModel(nn.Module):
        def __init__(self, in_dim=N_MELS, hidden=64, vocab=VOCAB_SIZE):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden * 2, vocab)

        def forward(self, x):
            h, _ = self.lstm(x)
            return F.log_softmax(self.fc(h), dim=-1)

    model = TinyCTCModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    losses, cers = [], []
    print(f"Training toy CTC model for {args.epochs} steps "
          f"(frames_per_char=({args.min_frames},{args.max_frames}))...")

    for step in range(args.epochs):
        word = random.choice(WORDS)
        frames = synthesize_frames(word)
        x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0)
        targets = torch.tensor([CHAR2IDX[c] for c in word], dtype=torch.long)

        log_probs = model(x).transpose(0, 1)  # (T, B, V)
        input_lengths = torch.tensor([log_probs.size(0)])
        target_lengths = torch.tensor([len(targets)])
        loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        with torch.no_grad():
            pred_ids = model(x).squeeze(0).argmax(dim=-1).tolist()
        pred_chars = [IDX2CHAR.get(i, BLANK) if i != 0 else BLANK for i in pred_ids]
        decoded = ctc_collapse(pred_chars)
        cers.append(edit_distance(decoded, word) / max(len(word), 1))

        if (step + 1) % 50 == 0:
            print(f"  step {step + 1:4d} | loss={np.mean(losses[-50:]):.4f} "
                  f"| CER(last 50)={np.mean(cers[-50:]) * 100:.1f}%")

    final_cer = float(np.mean(cers[-30:]) * 100)
    first_below_10 = next((i + 1 for i in range(0, len(cers), 10)
                            if np.mean(cers[i:i + 10]) < 0.10), None)

    print(f"\nFinal CER (last 30 steps): {final_cer:.1f}%")
    print(f"First 10-step window below 10% CER: step {first_below_10}")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"Model saved to {args.save}")

    os.makedirs('results', exist_ok=True)
    with open('results/ctc_results.json', 'w') as f:
        json.dump({'losses': losses, 'cers': cers, 'final_cer_pct': final_cer,
                   'first_below_10pct_step': first_below_10}, f)
    print("Metrics saved to results/ctc_results.json")


# ──────────────────────────────────────────────────────────────────────────
# Mode 2: --model wav2vec2-probe   (Part 4 / Exercise 3)
# ──────────────────────────────────────────────────────────────────────────

def run_wav2vec2_probe(args):
    """Extract frozen wav2vec2 features on a SpeechCommands subset, train a linear
    probe, compare against a raw mel-spectrogram baseline (Exercise 3a), and save
    a comparison bar chart + t-SNE plot of the wav2vec2 feature space."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchaudio.transforms as T
    from sklearn.model_selection import train_test_split
    from sklearn.manifold import TSNE
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if args.dataset != 'speechcommands':
        sys.exit(f"Unsupported --dataset '{args.dataset}'. Only 'speechcommands' is implemented.")

    probe_words = args.classes.split(',')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading pretrained wav2vec2 ({args.w2v_name})...")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.w2v_name)
    w2v_model = Wav2Vec2Model.from_pretrained(args.w2v_name).to(device).eval()
    for p in w2v_model.parameters():
        p.requires_grad = False

    # NOTE: torchaudio's SPEECHCOMMANDS downloader (download.tensorflow.org) is
    # frequently unreliable / slow to resume on stalled connections. We instead
    # stream the dataset from Hugging Face's parquet mirror with a shuffle buffer,
    # since the underlying data is grouped alphabetically by word and a plain
    # sequential scan can take a very long time to reach late-alphabet classes
    # like "yes" or "stop".
    from datasets import load_dataset
    print(f"Streaming SpeechCommands from Hugging Face (classes={probe_words})...")
    ds = load_dataset("google/speech_commands", split="train",
                       streaming=True, revision="refs/convert/parquet")
    ds = ds.shuffle(seed=42, buffer_size=10_000)
    label_names = ds.features['label'].names

    # The 'refs/convert/parquet' revision has a metadata bug: it declares
    # ClassLabel(num_classes=31), but some actual examples carry label=32.
    # Older `datasets` versions (e.g. 2.19.0) validate this strictly DURING
    # iteration itself (encode_nested_example -> ClassLabel.encode_example),
    # raising ValueError before our own bounds-check below ever runs. Casting
    # the column to a plain int64 Value strips that strict validation, while
    # `label_names` (captured above, BEFORE this cast) still lets us look up
    # the word string for any in-range label.
    from datasets import Value
    ds = ds.cast_column('label', Value('int64'))
    n_labels = len(label_names)

    by_label = {w: [] for w in probe_words}
    scanned, skipped_bad_label = 0, 0
    for example in ds:
        scanned += 1
        label_idx = example['label']
        if not (0 <= label_idx < n_labels):
            skipped_bad_label += 1
            continue
        label_str = label_names[label_idx]
        if label_str in by_label and len(by_label[label_str]) < args.n_per_class:
            wvf = torch.tensor(example['audio']['array'], dtype=torch.float32).unsqueeze(0)
            by_label[label_str].append(wvf)
        if all(len(v) >= args.n_per_class for v in by_label.values()):
            break
    print(f"Scanned {scanned} examples ({skipped_bad_label} had out-of-range labels).")
    for label, clips in by_label.items():
        print(f"  '{label}': {len(clips)} clips collected")

    print("Extracting frozen wav2vec2 features...")
    feats, labels_list = [], []
    with torch.no_grad():
        for label, clips in by_label.items():
            for wvf in clips:
                inputs = extractor(wvf.squeeze(0).numpy(), sampling_rate=16000,
                                    return_tensors='pt').to(device)
                out = w2v_model(**inputs).last_hidden_state
                pooled = out.mean(dim=1).squeeze(0).cpu()
                feats.append(pooled)
                labels_list.append(probe_words.index(label))

    X = torch.stack(feats)
    y = torch.tensor(labels_list)
    print(f"Extracted {X.shape[0]} clips, {X.shape[1]}-dim features, {len(probe_words)} classes")

    if not args.train:
        print("`--train` not passed; skipping linear-probe training.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), y.numpy(), test_size=0.3, random_state=42, stratify=y.numpy())
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    probe = nn.Linear(X.shape[1], len(probe_words))
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)

    for epoch in range(args.probe_epochs):
        logits = probe(X_train_t)
        loss = F.cross_entropy(logits, y_train_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        test_acc = (probe(X_test_t).argmax(1) == y_test_t).float().mean().item()

    random_baseline = 100 / len(probe_words)
    print(f"\nwav2vec2 linear probe test accuracy: {test_acc * 100:.1f}% "
          f"(random baseline: {random_baseline:.1f}%)")

    # --- Exercise 3(a): raw mel-spectrogram baseline, same protocol ---
    print("\nBuilding raw mel-spectrogram baseline for comparison...")
    mel_tf = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=320, n_mels=80)
    feats_mel, labels_mel = [], []
    for label, clips in by_label.items():
        for wvf in clips:
            mel = mel_tf(wvf)
            pooled = mel.mean(dim=-1).squeeze(0)
            feats_mel.append(pooled)
            labels_mel.append(probe_words.index(label))
    X_mel = torch.stack(feats_mel)
    y_mel = torch.tensor(labels_mel)

    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_mel.numpy(), y_mel.numpy(), test_size=0.3, random_state=42, stratify=y_mel.numpy())
    X_train_mt = torch.tensor(X_train_m, dtype=torch.float32)
    y_train_mt = torch.tensor(y_train_m, dtype=torch.long)
    X_test_mt = torch.tensor(X_test_m, dtype=torch.float32)
    y_test_mt = torch.tensor(y_test_m, dtype=torch.long)

    probe_mel = nn.Linear(X_mel.shape[1], len(probe_words))
    opt_mel = torch.optim.Adam(probe_mel.parameters(), lr=1e-2)
    for epoch in range(args.probe_epochs):
        logits = probe_mel(X_train_mt)
        loss = F.cross_entropy(logits, y_train_mt)
        opt_mel.zero_grad()
        loss.backward()
        opt_mel.step()
    with torch.no_grad():
        mel_test_acc = (probe_mel(X_test_mt).argmax(1) == y_test_mt).float().mean().item()

    print(f"Raw mel-spectrogram linear probe test accuracy: {mel_test_acc * 100:.1f}%")
    print(f"Gap (wav2vec2 - mel-spectrogram): {(test_acc - mel_test_acc) * 100:.1f} percentage points")

    # --- Visualization: comparison bar chart + t-SNE of wav2vec2 features ---
    os.makedirs('visualizations', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(['Mel-spectrogram\n(baseline)', 'wav2vec2\n(frozen)'],
                [mel_test_acc * 100, test_acc * 100], color=['#94a3b8', '#2563eb'])
    axes[0].axhline(random_baseline, color='red', linestyle='--', alpha=0.6,
                     label=f'Random baseline ({random_baseline:.1f}%)')
    axes[0].set_ylabel('Linear probe test accuracy (%)')
    axes[0].set_title('wav2vec2 vs. Mel-Spectrogram Linear Probe')
    axes[0].legend()
    for i, v in enumerate([mel_test_acc * 100, test_acc * 100]):
        axes[0].text(i, v + 1.5, f"{v:.1f}%", ha='center', fontweight='bold')

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(15, X.shape[0] - 1))
    X_2d = tsne.fit_transform(X.numpy())
    for i, word in enumerate(probe_words):
        mask = (y.numpy() == i)
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1], label=word, alpha=0.7)
    axes[1].set_title('t-SNE of Frozen wav2vec2 Features')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('visualizations/wav2vec2_vs_mel_and_tsne.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/wav2vec2_vs_mel_and_tsne.png")

    os.makedirs('results', exist_ok=True)
    with open('results/wav2vec2_probe_results.json', 'w') as f:
        json.dump({'classes': probe_words, 'n_per_class': args.n_per_class,
                   'wav2vec2_test_accuracy_pct': test_acc * 100,
                   'mel_spectrogram_test_accuracy_pct': mel_test_acc * 100,
                   'random_baseline_pct': random_baseline}, f)
    print("Metrics saved to results/wav2vec2_probe_results.json")


# ──────────────────────────────────────────────────────────────────────────
# Mode 3: --model voice-clone   (Part 5 / Exercise 4)
# ──────────────────────────────────────────────────────────────────────────

STYLE_TO_SE = {
    'us':    ('en-us.pth',    'EN-US'),
    'br':    ('en-br.pth',    'EN-BR'),
    'india': ('en-india.pth', 'EN_INDIA'),
    'au':    ('en-au.pth',    'EN-AU'),
}

LANGUAGE_TEXT_DEFAULTS = {
    'en': "Hello, this is a test of cross lingual voice cloning.",
    'es': "Hola, esta es una prueba de clonacion de voz entre idiomas.",
    'fr': "Bonjour, ceci est un test de clonage vocal interlingue.",
}


def _load_openvoice(device):
    """Shared setup: download checkpoints + build the ToneColorConverter."""
    from huggingface_hub import snapshot_download
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter

    ckpt_dir = snapshot_download(repo_id='myshell-ai/OpenVoiceV2')
    tone_color_converter = ToneColorConverter(f'{ckpt_dir}/converter/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_dir}/converter/checkpoint.pth')
    return ckpt_dir, se_extractor, tone_color_converter


def _hparams_to_dict(hp):
    """MeloTTS's `hps.data.spk2id` is sometimes a plain dict, sometimes wrapped in
    a custom HParams object that supports `.keys()`/`[...]` but not `.get()`.
    Normalize either case to a plain dict so the rest of the code can use `.get()`."""
    if isinstance(hp, dict):
        return hp
    if hasattr(hp, 'keys'):
        return {k: hp[k] for k in hp.keys()}
    return dict(vars(hp))


def run_voice_clone(args):
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('data/voice_clone', exist_ok=True)
    os.makedirs('data/voice_clone/processed', exist_ok=True)

    ckpt_dir, se_extractor, tone_color_converter = _load_openvoice(device)

    # --- Extract tone color embedding from a reference clip -----------------
    if args.extract_se:
        if not args.reference:
            sys.exit("--extract-se requires --reference <path-to-your-clip>")
        target_se, audio_name = se_extractor.get_se(
            args.reference, tone_color_converter,
            target_dir='data/voice_clone/processed', vad=True)
        se_path = 'data/voice_clone/my_tone_color_se.pth'
        torch.save(target_se, se_path)
        print(f"Extracted tone color embedding: shape {tuple(target_se.shape)}")
        print(f"Saved to {se_path}")
        return

    # Everything below (--generate) needs a previously extracted tone color
    se_path = 'data/voice_clone/my_tone_color_se.pth'
    if not os.path.exists(se_path):
        sys.exit(f"No saved tone color found at {se_path}. "
                  f"Run `--extract-se --reference <clip>` first.")
    target_se = torch.load(se_path, map_location=device)

    # NOTE: OpenVoiceV2 (unlike V1) does NOT ship a `base_speakers/config.json` /
    # `checkpoint.pth` pair, so there is no `BaseSpeakerTTS` to load here. In V2,
    # all base-speaker speech synthesis is delegated to MeloTTS instead, and
    # OpenVoice only supplies the tone-color converter + per-accent speaker
    # embeddings under `base_speakers/ses/*.pth`. We use MeloTTS for every
    # synthesis call below (both English accents and cross-lingual).
    from melo.api import TTS as MeloTTS

    # --- Cross-lingual cloning ----------------------------------------------
    if args.language:
        lang = args.language.lower()
        text = args.text or LANGUAGE_TEXT_DEFAULTS.get(lang, "Hello.")
        base_tts = MeloTTS(language=lang.upper(), device=device)
        spk_ids = _hparams_to_dict(base_tts.hps.data.spk2id)
        spk_key = list(spk_ids.keys())[0]

        base_path = f'data/voice_clone/base_{lang}.wav'
        out_path = f'data/voice_clone/cloned_{lang}.wav'
        base_tts.tts_to_file(text, spk_ids[spk_key], base_path)

        # OpenVoiceV2's checkpoint only ships `ses/` files for the 4 English
        # accents; non-English base speakers fall back to EN-Default as the
        # closest available source SE (V2 provides no per-language source SE).
        default_se_path = f'{ckpt_dir}/base_speakers/ses/en-default.pth'
        source_se = torch.load(default_se_path, map_location=device)
        tone_color_converter.convert(audio_src_path=base_path, src_se=source_se,
                                      tgt_se=target_se, output_path=out_path, tau=0.3)
        print(f"[{lang}] '{text}' -> {out_path}")
        return

    # --- Single accent or all accents ---------------------------------------
    if not args.text:
        sys.exit("--generate requires --text \"...\"")

    # One English MeloTTS model covers all 4 accents (EN-US, EN-BR, EN_INDIA,
    # EN-AU) via its spk2id map -- matches the keys used in STYLE_TO_SE below.
    en_tts = MeloTTS(language='EN', device=device)
    spk_ids = _hparams_to_dict(en_tts.hps.data.spk2id)

    accents_to_run = list(STYLE_TO_SE.keys()) if args.accent == 'all' else [args.accent]
    generated_paths = {}
    for accent in accents_to_run:
        if accent not in STYLE_TO_SE:
            print(f"Skipping unknown accent '{accent}' (known: {list(STYLE_TO_SE.keys())})")
            continue
        se_file, spk_key = STYLE_TO_SE[accent]
        spk_id = spk_ids.get(spk_key, spk_ids.get('EN-US'))

        base_path = f'data/voice_clone/base_{accent}.wav'
        out_path = f'data/voice_clone/cloned_{accent}.wav'
        en_tts.tts_to_file(args.text, spk_id, base_path, speed=1.0)

        source_se = torch.load(f'{ckpt_dir}/base_speakers/ses/{se_file}', map_location=device)
        tone_color_converter.convert(audio_src_path=base_path, src_se=source_se,
                                      tgt_se=target_se, output_path=out_path, tau=0.3)
        print(f"[{accent:8}] '{args.text}' -> {out_path}")
        generated_paths[accent] = out_path

    # --- Visualization (Part 5.4): mel-spectrogram grid across all 4 accents,
    # plus cosine-similarity check between each generated clip and the reference
    # tone color (Exercise 4b) -- only meaningful when all 4 accents were just
    # generated together. ---
    if args.accent == 'all' and len(generated_paths) == len(STYLE_TO_SE):
        import torchaudio
        import torchaudio.transforms as T
        import torch.nn.functional as F
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs('visualizations', exist_ok=True)
        mel_tf = T.MelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80)

        fig, axes = plt.subplots(1, len(generated_paths), figsize=(18, 4))
        similarities = {}
        for ax, (accent, path) in zip(axes, generated_paths.items()):
            wvf, sr = torchaudio.load(path)
            if sr != 22050:
                wvf = T.Resample(sr, 22050)(wvf)
            mel = mel_tf(wvf).log2()[0].numpy()
            ax.imshow(mel, origin='lower', aspect='auto', cmap='magma')
            ax.set_title(accent)
            ax.set_xlabel('Frame')
            if ax is axes[0]:
                ax.set_ylabel('Mel bin')

            # Exercise 4(b): cosine similarity vs reference tone color
            se_generated, _ = se_extractor.get_se(
                path, tone_color_converter, target_dir='data/voice_clone/processed', vad=True)
            sim = F.cosine_similarity(target_se.flatten().unsqueeze(0),
                                       se_generated.flatten().unsqueeze(0)).item()
            similarities[accent] = sim

        plt.suptitle('Mel Spectrograms: Same Cloned Voice Across 4 Accents')
        plt.tight_layout()
        plt.savefig('visualizations/mel_spectrogram_grid_accents.png', dpi=150, bbox_inches='tight')
        print("\nSaved: visualizations/mel_spectrogram_grid_accents.png")

        print("\nCosine similarity to reference tone color (Exercise 4b):")
        for accent, sim in similarities.items():
            print(f"  {accent:8}: {sim:.4f}")
        print(f"  Mean: {np.mean(list(similarities.values())):.4f}  "
              f"Std: {np.std(list(similarities.values())):.4f}")

        os.makedirs('results', exist_ok=True)
        with open('results/voice_clone_results.json', 'w') as f:
            json.dump({'text': args.text, 'cosine_similarities': similarities}, f)
        print("Metrics saved to results/voice_clone_results.json")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="A6 Speech Processing — unified CLI")
    p.add_argument('--model', required=True,
                   choices=['ctc', 'wav2vec2-probe', 'voice-clone'])

    # --- ctc ---
    p.add_argument('--epochs', type=int, default=300, help='[ctc] training steps')
    p.add_argument('--train', action='store_true', help='[ctc, wav2vec2-probe] run training')
    p.add_argument('--min-frames', dest='min_frames', type=int, default=3,
                   help='[ctc] min frames per character')
    p.add_argument('--max-frames', dest='max_frames', type=int, default=8,
                   help='[ctc] max frames per character')
    p.add_argument('--seed', type=int, default=42, help='[ctc] random seed')
    p.add_argument('--save', type=str, default=None, help='[ctc] path to save model weights')

    # --- wav2vec2-probe ---
    p.add_argument('--dataset', type=str, default='speechcommands',
                   help='[wav2vec2-probe] dataset name')
    p.add_argument('--classes', type=str, default='yes,no,stop,go',
                   help='[wav2vec2-probe] comma-separated class list')
    p.add_argument('--n-per-class', dest='n_per_class', type=int, default=40,
                   help='[wav2vec2-probe] clips per class')
    p.add_argument('--w2v-name', dest='w2v_name', type=str, default='facebook/wav2vec2-base',
                   help='[wav2vec2-probe] HF model name')
    p.add_argument('--probe-epochs', dest='probe_epochs', type=int, default=100,
                   help='[wav2vec2-probe] linear probe training epochs')

    # --- voice-clone ---
    p.add_argument('--extract-se', action='store_true',
                   help='[voice-clone] extract tone color from --reference')
    p.add_argument('--reference', type=str, default=None,
                   help='[voice-clone] path to reference voice clip')
    p.add_argument('--accent', type=str, default=None,
                   help="[voice-clone] 'us' | 'br' | 'india' | 'au' | 'all'")
    p.add_argument('--language', type=str, default=None,
                   help="[voice-clone] cross-lingual target, e.g. 'es', 'fr'")
    p.add_argument('--text', type=str, default=None, help='[voice-clone] text to synthesize')
    p.add_argument('--generate', action='store_true',
                   help='[voice-clone] synthesize using a previously extracted tone color')

    return p


def main():
    args = build_parser().parse_args()

    if args.model == 'ctc':
        run_ctc(args)
    elif args.model == 'wav2vec2-probe':
        run_wav2vec2_probe(args)
    elif args.model == 'voice-clone':
        run_voice_clone(args)


if __name__ == '__main__':
    main()