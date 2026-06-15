# A3: Self-Supervised Learning

## Project Overview

This project is part of the Deep Learning course assignment on **Self-Supervised Learning**. The main goal is to implement and compare different self-supervised learning methods on the CIFAR-10 dataset.

The assignment focuses on three self-supervised learning approaches:

1. **DINO** — Self-distillation with no labels
2. **MAE** — Masked Autoencoder
3. **SimCLR** — Contrastive learning baseline / comparison

Currently, **Exercise 1: DINO ablation study** has been completed.

---

# Environment

The experiments were run using the following environment:

```text
Python: 3.10.20
PyTorch: 2.6.0+cu124
Torchvision: 0.21.0+cu124
GPU: NVIDIA RTX A6000
Dataset: CIFAR-10
```

---

# Project Structure

```text
A3/
├── README.md
├── requirements.txt
├── run.py
├── saved/
│   ├── dino_default.pt
│   ├── dino_no_centering.pt
│   └── dino_no_local.pt
├── results/
│   ├── dino_default_loss.png
│   ├── dino_default_center_norm.png
│   ├── dino_default_linear.txt
│   ├── dino_no_centering_loss.png
│   ├── dino_no_centering_center_norm.png
│   ├── dino_no_centering_linear.txt
│   ├── dino_no_local_loss.png
│   ├── dino_no_local_center_norm.png
│   └── dino_no_local_linear.txt
├── notebooks/
├── src/
└── configs/
```

---

# Exercise 1: DINO Ablation Study

## Objective

The objective of Exercise 1 is to study the effect of two important components of DINO:

1. **Centering**
2. **Local crops**

DINO is a self-supervised learning method where a student network learns to match the output of a teacher network using different augmented views of the same image. The teacher network is updated using exponential moving average of the student network.

In this exercise, three DINO variants were trained and evaluated:

| Setting        | Description                                       |
| -------------- | ------------------------------------------------- |
| Default DINO   | Uses 2 global crops, 4 local crops, and centering |
| No centering   | Removes the centering mechanism from DINO loss    |
| No local crops | Uses only 2 global crops and removes local crops  |

After self-supervised pretraining, each frozen encoder was evaluated using **linear evaluation** on CIFAR-10.

---

## Training Commands

### 1. Default DINO

```bash
python run.py --model dino --epochs 50 --batch-size 256 --train
```

### 2. DINO without centering

```bash
python run.py --model dino --no-centering --epochs 50 --batch-size 256 --train
```

### 3. DINO without local crops

```bash
python run.py --model dino --n-local 0 --epochs 50 --batch-size 256 --train
```

---

## Linear Evaluation Commands

### 1. Default DINO

```bash
python run.py --model dino --weights saved/dino_default.pt --evaluate --linear --linear-epochs 20 --batch-size 256
```

### 2. DINO without centering

```bash
python run.py --model dino --weights saved/dino_no_centering.pt --evaluate --linear --linear-epochs 20 --batch-size 256
```

### 3. DINO without local crops

```bash
python run.py --model dino --weights saved/dino_no_local.pt --evaluate --linear --linear-epochs 20 --batch-size 256
```

---

# Exercise 1 Results

## Linear Evaluation Accuracy

| DINO Setting   | Linear Eval Accuracy |
| -------------- | -------------------: |
| Default DINO   |           **63.08%** |
| No centering   |           **36.76%** |
| No local crops |           **60.25%** |

---

# Discussion of Exercise 1

The default DINO model achieved the best linear evaluation accuracy of **63.08%**. This setting uses both centering and local crops, which are important components of DINO training.

Removing centering reduced the accuracy to **36.76%**. This shows that centering is very important for stabilizing the teacher output distribution. Without centering, the teacher output may become biased toward some dimensions, and the student may learn weaker or collapsed representations.

Removing local crops achieved **60.25%**, which is slightly lower than the default DINO result. This shows that local crops help the model learn better visual representations by forcing the student to match small local image regions with global image views. However, the drop is not as large as the no-centering experiment, meaning the model can still learn useful features using only global crops.

Overall, the results show that **centering is more critical than local crops** in this experiment. Local crops improve representation quality, but removing centering causes a much larger performance drop.

---

# DINO Center Norm

The center norm was saved during training for each DINO setting. The center norm helps track how the DINO center vector changes during training.

For the default DINO model, the center norm stabilized near the end of training:

```text
Epoch 050/50 | Loss: 1.7499 | Center norm: 0.4998
```

This indicates that the teacher output distribution became more stable over training.

---

# Saved Results

The following result files were generated:

```text
results/dino_default_loss.png
results/dino_default_center_norm.png
results/dino_default_linear.txt

results/dino_no_centering_loss.png
results/dino_no_centering_center_norm.png
results/dino_no_centering_linear.txt

results/dino_no_local_loss.png
results/dino_no_local_center_norm.png
results/dino_no_local_linear.txt
```

---

# Exercise 1 Conclusion

Exercise 1 shows that DINO can learn useful image representations without using labels. The best result was obtained using the full default DINO setup with centering and local crops. The no-centering experiment performed much worse, showing that centering is essential for preventing weak or unstable representations. The no-local-crops experiment performed slightly worse than default DINO, showing that local crops improve feature learning but are less critical than centering.

---


# Exercise 2: MAE Mask Ratio Study

## Objective

The objective of Exercise 2 is to study the effect of different mask ratios in Masked Autoencoder (MAE) training. MAE learns image representations by masking part of an image and training the model to reconstruct the missing patches.

In this experiment, three mask ratios were tested:

| Mask Ratio | Description                     |
| ---------: | ------------------------------- |
|       0.25 | 25% of image patches are masked |
|       0.50 | 50% of image patches are masked |
|       0.75 | 75% of image patches are masked |

After self-supervised MAE pretraining, the encoder was frozen and evaluated using linear evaluation on CIFAR-10.

---

## Training Commands

### MAE with 0.25 mask ratio

```bash
python -u run.py --model mae --mask-ratio 0.25 --epochs 50 --batch-size 256 --train | tee logs/mae_025_train.log
```

### MAE with 0.50 mask ratio

```bash
python -u run.py --model mae --mask-ratio 0.50 --epochs 50 --batch-size 256 --train | tee logs/mae_050_train.log
```

### MAE with 0.75 mask ratio

```bash
python -u run.py --model mae --mask-ratio 0.75 --epochs 50 --batch-size 256 --train | tee logs/mae_075_train.log
```

---

## Linear Evaluation Commands

### Linear evaluation for 0.25 mask ratio

```bash
python -u run.py --model mae --mask-ratio 0.25 --weights saved/mae_encoder_025.pt --evaluate --linear --linear-epochs 20 --batch-size 256 | tee logs/mae_025_linear.log
```

### Linear evaluation for 0.50 mask ratio

```bash
python -u run.py --model mae --mask-ratio 0.50 --weights saved/mae_encoder_050.pt --evaluate --linear --linear-epochs 20 --batch-size 256 | tee logs/mae_050_linear.log
```

### Linear evaluation for 0.75 mask ratio

```bash
python -u run.py --model mae --mask-ratio 0.75 --weights saved/mae_encoder_075.pt --evaluate --linear --linear-epochs 20 --batch-size 256 | tee logs/mae_075_linear.log
```

---

## Exercise 2 Results

| Mask Ratio | Final Reconstruction Loss | Linear Eval Accuracy |
| ---------: | ------------------------: | -------------------: |
|       0.25 |                **0.0777** |           **50.49%** |
|       0.50 |                **0.1251** |           **52.33%** |
|       0.75 |                **0.2338** |           **54.05%** |

---

## Discussion

The reconstruction loss increased as the mask ratio increased. This is expected because a higher mask ratio makes the reconstruction task more difficult. When only 25% of patches are masked, the model can reconstruct the image more easily because most of the image is still visible. Therefore, the 0.25 mask ratio produced the lowest reconstruction loss of **0.0777**.

However, the best linear evaluation accuracy was achieved with the 0.75 mask ratio, which reached **54.05%**. This shows that lower reconstruction loss does not always mean better representation learning. With a low mask ratio, the model may solve the reconstruction task using nearby visible patches without learning strong semantic features. With a high mask ratio, the model is forced to understand more global image structure because most of the image is hidden.

The 0.75 mask ratio produced the highest reconstruction loss but also the best downstream classification accuracy. This suggests that harder reconstruction tasks can encourage the encoder to learn more useful visual representations.

---

## Saved MAE Results

The following files were generated:

```text
saved/mae_encoder_025.pt
saved/mae_encoder_050.pt
saved/mae_encoder_075.pt

results/mae_encoder_025_loss.png
results/mae_encoder_050_loss.png
results/mae_encoder_075_loss.png

results/mae_encoder_025_reconstruction.png
results/mae_encoder_050_reconstruction.png
results/mae_encoder_075_reconstruction.png

results/mae_encoder_025_linear.txt
results/mae_encoder_050_linear.txt
results/mae_encoder_075_linear.txt
```

---

## Exercise 2 Conclusion

Exercise 2 shows that MAE representation quality depends strongly on the mask ratio. A lower mask ratio gives better reconstruction loss because the task is easier, but it does not necessarily produce the best representation. In this experiment, the 0.75 mask ratio gave the best linear evaluation accuracy, showing that harder masking encouraged the model to learn stronger and more transferable features.



# Exercise 3: SimCLR vs DINO vs MAE Comparison

## Objective

The objective of Exercise 3 is to compare three self-supervised learning methods on CIFAR-10:

1. **SimCLR** — contrastive learning with negative pairs
2. **DINO** — self-distillation using an EMA teacher
3. **MAE** — masked image reconstruction

Each method was pretrained without labels. After pretraining, the encoder was frozen and evaluated using linear evaluation on CIFAR-10.

---

## Training Commands

### SimCLR Training

```bash
python -u run_simclr.py --epochs 50 --batch-size 256 --train | tee logs/simclr_train.log
```

### SimCLR Linear Evaluation

```bash
python -u run_simclr.py --weights saved/simclr.pt --evaluate --linear --linear-epochs 20 --batch-size 256 | tee logs/simclr_linear.log
```

---

## Exercise 3 Results

| Method | Final Training Loss | Linear Eval Accuracy | Best Linear Eval Accuracy | Time per Epoch |
| ------ | ------------------: | -------------------: | ------------------------: | -------------: |
| SimCLR |          **4.7899** |           **64.88%** |                **64.97%** |         ~13.6s |
| DINO   |          **1.7499** |           **63.08%** |                **63.08%** |         ~41.5s |
| MAE    |          **0.2338** |           **54.05%** |                **54.14%** |          ~5.0s |

---

## Method Comparison

| Feature                   | SimCLR                                                     | DINO                                   | MAE                                |
| ------------------------- | ---------------------------------------------------------- | -------------------------------------- | ---------------------------------- |
| Learning type             | Contrastive learning                                       | Self-distillation                      | Masked image reconstruction        |
| Uses labels?              | No                                                         | No                                     | No                                 |
| Needs negative pairs?     | Yes                                                        | No                                     | No                                 |
| Uses EMA teacher?         | No                                                         | Yes                                    | No                                 |
| Main objective            | Pull positive pairs together and push negative pairs apart | Match student output to teacher output | Reconstruct masked image patches   |
| Best linear eval accuracy | **64.97%**                                                 | **63.08%**                             | **54.14%**                         |
| Training speed            | Medium                                                     | Slowest                                | Fastest                            |
| Saved visual output       | Loss curve                                                 | Loss curve and center norm             | Loss curve and reconstruction grid |

---

## Discussion

In this experiment, **SimCLR achieved the best linear evaluation accuracy**, with a best accuracy of **64.97%**. This shows that contrastive learning was very effective for CIFAR-10 in this setup. SimCLR learns by creating two augmented views of the same image, treating them as a positive pair, and pushing representations of other images away as negative pairs.

**DINO achieved 63.08%**, which is slightly lower than SimCLR. DINO does not require negative pairs. Instead, it uses a student network and a teacher network. The teacher is updated using exponential moving average of the student. DINO also uses centering and multi-crop augmentation to stabilize training and improve representation learning.

**MAE achieved 54.05% final accuracy**, which is lower than SimCLR and DINO. However, MAE trained much faster in this experiment. MAE learns by masking image patches and reconstructing the missing parts. The best MAE result came from the 0.75 mask ratio, showing that harder reconstruction encouraged better representation learning.

The training losses are not directly comparable across methods because each method uses a different objective. SimCLR uses contrastive loss, DINO uses self-distillation loss, and MAE uses reconstruction loss. Therefore, linear evaluation accuracy is the most meaningful metric for comparing representation quality.

---

## Why MAE is Popular for Large-Scale Pretraining

MAE is popular for large-scale pretraining because it is simple, scalable, and efficient. Since most image patches are masked, the encoder only processes visible patches, which can reduce computation. MAE also does not require labels, negative pairs, or a teacher network. This makes it suitable for training on very large image datasets.

However, MAE focuses mainly on reconstruction. For small datasets or fine-grained classification tasks, reconstruction quality does not always guarantee the best semantic representation.

---

## Why DINO is Useful for Segmentation

DINO is often useful for segmentation because it learns strong visual attention and object-level representations. Since DINO uses self-distillation and multi-crop training, it can learn both global and local image structure. This makes DINO features useful for dense prediction tasks such as object localization and segmentation.

In this experiment, removing local crops reduced DINO accuracy from 63.08% to 60.25%, showing that local crops help the model learn better local visual features.

---

## Which Method for Medical Image Segmentation with 500 Scans?

For a medical image segmentation task with only 500 scans, **DINO would be the preferred choice** among these three methods.

The reason is that medical datasets are usually small, and segmentation requires strong local and spatial representations. DINO can learn meaningful object-level and local features using multi-crop training and teacher-student self-distillation. It also does not require negative pairs, which can be difficult to define correctly in medical datasets where many images may look similar.

MAE is also useful, especially if there are many unlabeled medical images available. However, with only 500 scans, MAE may focus too much on low-level reconstruction rather than learning semantic boundaries. SimCLR can work well, but it depends strongly on good augmentation design and negative pairs. In medical imaging, aggressive augmentations or incorrect negative assumptions may hurt performance.

Therefore, for medical image segmentation with a small dataset, DINO is a strong choice because it can learn useful local features and attention-like representations without requiring labels or negative pairs.

---

## Exercise 3 Conclusion

Exercise 3 shows that all three self-supervised learning methods learned useful representations from CIFAR-10. SimCLR achieved the best linear evaluation accuracy, DINO performed very close to SimCLR and provided useful teacher-student representation learning, while MAE trained fastest and produced reconstruction visualizations.

Overall, the results suggest that SimCLR was the strongest method for CIFAR-10 classification in this experiment, while DINO may be more suitable for segmentation-style tasks because of its local feature learning and self-distillation behavior.
