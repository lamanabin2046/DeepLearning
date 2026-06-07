# A2-02: Image Segmentation with U-Net

## Overview

This assignment implements semantic image segmentation using U-Net with a pretrained ResNet-18 encoder on the Oxford-IIIT Pet dataset. The main experiment compares U-Net with and without skip connections to understand their importance in segmentation.

---

## Dataset

- **Dataset**: Oxford-IIIT Pet Dataset
- **Train**: 3680 images
- **Test**: 3669 images
- **Classes**: 3 (Pet, Background, Border)
- **Image size**: 128x128

---

## Model Architecture

Both models use the same pretrained ResNet-18 encoder (ImageNet weights). The only difference is whether the decoder uses skip connections or not.

```
Input (128x128x3)
      |
ResNet-18 Encoder
  enc0: 64x64x64
  enc1: 32x32x64
  enc2: 16x16x128
  enc3: 8x8x256
  enc4: 4x4x512  <- bottleneck
      |
Decoder (with or without skip connections)
      |
Output Mask (128x128x3)
```

---

## Commands Used

```bash
# Train with skip connections (baseline)
python3 run.py --model unet_resnet18 --dataset oxford_pet --epochs 20 --train

# Train without skip connections (ablation)
python3 run.py --model unet_resnet18_no_skip --dataset oxford_pet --epochs 20 --train

# Evaluate unet_resnet18
python3 run.py --model unet_resnet18 --weights unet_resnet18_pet.pt --dataset oxford_pet --evaluate

# Evaluate unet_resnet18_no_skip
python3 run.py --model unet_resnet18_no_skip --weights unet_resnet18_no_skip_pet.pt --dataset oxford_pet --evaluate
```

---

## Results Table

| Model | Encoder | Skip connections | Val mIoU | Time/epoch | Params |
|---|---|---|---|---|---|
| unet_resnet18 | ResNet-18 (ImageNet) | Yes | 0.7608 | ~19.8s | 14,343,491 |
| unet_resnet18_no_skip | ResNet-18 (ImageNet) | No | 0.6942 | ~19.5s | 13,532,483 |

---

## Exercise Answers

### c) Why do skip connections help segmentation more than classification?

In classification, the model only needs to output one label for the whole image so spatial details do not matter. The encoder can compress the image aggressively and still produce the correct answer. In segmentation, the model must predict a label for every single pixel, which requires knowing exactly where objects are in the image. Fine spatial details like edges and boundaries are critical. Skip connections pass high-resolution feature maps directly from the encoder to the decoder, allowing the model to recover spatial details that are lost during pooling. Without skip connections, the decoder must reconstruct these details from a very compressed bottleneck, which is much harder and leads to blurry boundaries.

### d) Which skip connection level hurts most when removed?

The first skip connection (64ch, highest resolution) hurts the most when removed. This layer captures low-level features like edges, corners, and fine boundaries at high resolution (64x64). These fine details are critical for precise pixel-level segmentation, especially at object boundaries. The last skip connection (512ch, lowest resolution) captures high-level semantic features at very low resolution (8x8), which the decoder can partially reconstruct from the bottleneck since both contain similar semantic information. Therefore, removing the first skip connection causes the greatest loss in boundary precision.

---

## Discussion

Skip connections improved mIoU from 0.6942 to 0.7608, a gain of 0.0666 (9.6% relative improvement). This demonstrates that skip connections are critical for recovering fine spatial details in segmentation tasks. The model without skip connections struggled especially in the early epochs, starting at mIoU of 0.4928 compared to 0.6970 for the model with skip connections, showing that skip connections also speed up convergence significantly.

U-Net would be chosen over Mask R-CNN when the task requires labeling every pixel in the image without needing to distinguish individual object instances, such as medical image segmentation, road scene parsing, or satellite image analysis. Mask R-CNN is better suited when you need to count and separately identify individual objects of the same class, such as counting people in a crowd or detecting multiple overlapping instances. U-Net is also preferred when computational resources are limited, as it is simpler and faster to train than Mask R-CNN.

---

## Repository Structure

```
A2-02/
├── run.py                              <- training and evaluation script
├── README.md                           <- this file
├── data/                               <- Oxford Pet dataset (auto downloaded)
├── logs/
│   ├── train_unet_resnet18.txt
│   ├── train_unet_resnet18_no_skip.txt
│   ├── eval_unet_resnet18.txt
│   └── eval_unet_resnet18_no_skip.txt
├── unet_resnet18_results.png           <- visualization
├── unet_resnet18_no_skip_results.png   <- visualization
├── unet_resnet18_training.png          <- loss and mIoU plot
└── unet_resnet18_no_skip_training.png  <- loss and mIoU plot
```