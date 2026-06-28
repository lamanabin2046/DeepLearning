# A6: Speech Processing

Implementation and evaluation of speech tokenization, CTC-based alignment, wav2vec2 self-supervised features, and OpenVoice voice cloning. All results below were measured by running `run.py` directly (see Commands Used).

## Setup

```bash
conda create -n a6-speech python=3.10 -y
conda activate a6-speech
pip install -r requirements.txt

# OpenVoice + MeloTTS (installed from source, not PyPI)
git clone https://github.com/myshell-ai/OpenVoice.git && cd OpenVoice && pip install -e . && cd ..
git clone https://github.com/myshell-ai/MeloTTS.git  && cd MeloTTS  && pip install -e . && cd ..
python -m unidic download
```

## Commands Used

```bash
# Exercise 2: train the toy CTC model, track character error rate
python3 run.py --model ctc --epochs 300 --train

# Exercise 3: linear-probe a frozen wav2vec2 checkpoint vs. a raw mel-spectrogram baseline
python3 run.py --model wav2vec2-probe --dataset speechcommands --classes yes,no,stop,go --train

# Exercise 4: extract a tone color embedding from a personal reference recording
python3 run.py --model voice-clone --extract-se --reference my_voice.wav

# Exercise 4: synthesize the cloned voice in a single accent
python3 run.py --model voice-clone --accent us --text "I got the job!" --generate

# Exercise 4: synthesize all four accents + compute cosine similarity to the reference tone color
python3 run.py --model voice-clone --accent all --text "I got the job! I am so excited about this new opportunity and I can not wait to get started." --generate

# Exercise 4: cross-lingual cloning
python3 run.py --model voice-clone --language es --text "Hola, como estas?" --generate
```

## Results

| Task | Model / Method | Result | Notes |
|---|---|---|---|
| Tokenization (Ex 1) | SpeechTokenizer | — | Word tokens stay flat (4-5) regardless of sentence length; character tokens scale with text and shift with normalization (e.g. `"Dr."` → `"doctor"` adds 3 chars). See `visualizations/tokenization_comparison.png`. |
| CTC character error rate (Ex 2) | Toy BiLSTM + CTC | **0.7% final CER** | Dropped below 10% CER by step ~71-80 of 300. Counter-intuitively, shorter character durations `(1,2)` converged *faster* than the default `(3,8)` (first <10% by step 30 vs. ~80-120), consistent across 3 seeds. |
| wav2vec2 vs. raw-feature probe (Ex 3) | Linear probe (frozen features) | **85.4% vs. 52.1%** | wav2vec2 beat the raw mel-spectrogram baseline by a 33.3-point margin, despite never being trained to classify words. Margin above random baseline held up well scaling from 4 to 8 classes (measured separately: 62.5pt → 60.4pt margin). For comparison, my A3 (SSL) lab's best result (SimCLR, 64.97% linear-eval on CIFAR-10) vs. a general-reference raw-pixel-MLP baseline (~35-40%, no exact number measured in A3) gives an estimated vision gap of ~25-30 points — similar order of magnitude to the 33.3-point speech gap, though the two aren't directly comparable (different objectives, pretraining scale, and task difficulty). |
| Voice cloning: accent + cross-lingual (Ex 4) | OpenVoice v2 + MeloTTS | **0.858 mean cosine sim** (std 0.014) | Tone color similarity to the reference recording stayed high and tightly clustered across all 4 accents (us: 0.859, br: 0.869, india: 0.836, au: 0.870) — indicates OpenVoice's identity/accent disentanglement is working as intended. |

## Visualizations

| File | Description |
|---|---|
| `visualizations/tokenization_comparison.png` | NLP word tokens vs. raw characters vs. speech tokenizer tokens, across the 5 test sentences |
| `visualizations/ctc_grid_and_cer_curve.png` | Greedy CTC decoding examples + character error rate curve vs. training step |
| `visualizations/wav2vec2_vs_mel_and_tsne.png` | wav2vec2 vs. mel-spectrogram linear probe accuracy, plus a t-SNE projection of the wav2vec2 feature space |
| `visualizations/mel_spectrogram_grid_accents.png` | Mel spectrograms of the same cloned voice synthesized in all 4 accents |

## Discussion

Working through tokenization and CTC back to back makes clear that speech and text are fundamentally different units to model: text tokenization is a largely solved, deterministic mapping from symbols to IDs, but speech has no fixed boundary between "frames" and "characters" — the same word can be spread over wildly different numbers of frames depending on speaking rate, which is exactly the alignment problem CTC's blank token and collapse rule exist to solve. This reframes how I'd think about training a TTS or ASR model going forward: text normalization (expanding `"Dr."`, spelling out digits) has to happen *before* tokenization because the model only ever sees characters, never abbreviation conventions, while on the acoustic side, the model needs an explicit mechanism (CTC's blank, or attention-based alignment) to handle the fact that input and output sequences are different, variable lengths with no natural one-to-one correspondence. A tone color embedding is a fundamentally different kind of conditioning signal than either a text token or a CTC blank: a text token (or an accent tag like `[EN-US]`) is a discrete symbol from a small fixed vocabulary that conditions *content or style* through attention, and a CTC blank is a structural placeholder that exists purely to make repeated-vs-continued symbols unambiguous during decoding — neither carries continuous, speaker-specific identity information. A tone color embedding, by contrast, is a dense, continuous vector extracted directly from a reference waveform that represents *who is speaking*, independent of what is being said or how it's said stylistically; that's precisely why it can be swapped onto four different MeloTTS accent outputs in this assignment and still preserve a consistent, measurable similarity (cosine sim 0.836-0.870) to the original speaker, something neither a token ID nor a blank symbol is designed to represent.

## Repository Structure

```
.
├── run.py                  # CLI entry point: ctc | wav2vec2-probe | voice-clone
├── requirements.txt
├── README.md
├── visualizations/          # 4 required result images
├── results/                 # JSON metrics auto-saved by run.py
└── data/
    ├── speechcommands/
    └── voice_clone/
```
