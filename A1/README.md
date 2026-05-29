# Assignment 1 – Representation Learning

**Course:** Deep Learning
**Dataset:** CIFAR-10

---

# Repository Structure

```text
A1/
├── alexnet.py
├── googlenet.py
├── resnet.py
├── run.py
├── models/
├── plots/
├── logs/
└── README.md
```

* `alexnet.py` contains the AlexNet implementation.
* `googlenet.py` contains the GoogLeNet (Inception) implementation.
* `resnet.py` contains the ResNet-18 implementation.
* `run.py` contains the training and testing pipeline.
* `models/` stores trained model weights.
* `plots/` stores training curves.
* `logs/` stores training and testing logs.

---

# Exercise 1 – Question 1

## Objective

Implement and train three neural network architectures on the CIFAR-10 dataset:

1. AlexNet
2. GoogLeNet
3. ResNet-18

A common training framework was developed in `run.py` to support training, validation, testing, checkpoint saving, and plotting.

---

## AlexNet

AlexNet was implemented using five convolutional layers followed by three fully connected layers. CIFAR-10 images were resized to 224×224 to match the architecture used in the professor's notebook.

### Architecture

* Conv(3 → 64)
* Conv(64 → 192)
* Conv(192 → 384)
* Conv(384 → 256)
* Conv(256 → 256)
* Fully Connected Layers (4096 → 4096 → 10)

### Results

| Metric                   |       Value |
| ------------------------ | ----------: |
| Parameters               |  57,044,810 |
| Epochs                   |          10 |
| Best Validation Accuracy |      62.86% |
| Test Accuracy            |      63.14% |
| Average Time / Epoch     | ~53 seconds |

---

## GoogLeNet

GoogLeNet was implemented using the CIFAR-10 version from the professor's notebook. The model uses Inception modules and accepts 32×32 input images. This implementation does not include auxiliary classifiers.

### Architecture

* Initial convolution layer
* Inception modules
* Global Average Pooling
* Fully Connected Output Layer

### Results

| Metric                   |       Value |
| ------------------------ | ----------: |
| Parameters               |   6,166,250 |
| Epochs                   |          25 |
| Best Validation Accuracy |      85.90% |
| Test Accuracy            |      85.97% |
| Average Time / Epoch     | ~61 seconds |

---

## ResNet-18

ResNet-18 was implemented using residual blocks with skip connections. A CIFAR-10 adapted 3×3 convolution stem was used instead of the original 7×7 ImageNet stem.

### Architecture

* Initial 3×3 convolution
* Four residual stages
* Global Average Pooling
* Fully Connected Output Layer

### Results

| Metric                   |       Value |
| ------------------------ | ----------: |
| Parameters               |  11,173,962 |
| Epochs                   |          20 |
| Best Validation Accuracy |      77.62% |
| Test Accuracy            |      78.33% |
| Average Time / Epoch     | ~17 seconds |

---

## Overall Comparison

| Model     | Parameters | Validation Accuracy | Test Accuracy |
| --------- | ---------: | ------------------: | ------------: |
| AlexNet   | 57,044,810 |              62.86% |        63.14% |
| GoogLeNet |  6,166,250 |              85.90% |        85.97% |
| ResNet-18 | 11,173,962 |              77.62% |        78.33% |

### Discussion

Among the three models, GoogLeNet achieved the highest accuracy while using significantly fewer parameters than AlexNet. The Inception architecture enables efficient multi-scale feature extraction and provides excellent parameter efficiency.

AlexNet contained approximately 57 million parameters due to its large fully connected layers, yet achieved the lowest accuracy among the three models. This demonstrates that a larger parameter count does not necessarily lead to better performance.

ResNet-18 achieved a good balance between accuracy and efficiency. The residual connections helped optimization and enabled deeper feature learning while maintaining relatively fast training speed.

Overall, GoogLeNet produced the best performance on CIFAR-10 in terms of classification accuracy and parameter efficiency.

---

# Training Commands

## AlexNet

```bash
python3 run.py --model alexnet --dataset cifar10 --epochs 10 --batch_size 64 --train
```

## GoogLeNet

```bash
python3 run.py --model googlenet --dataset cifar10 --epochs 25 --batch_size 64 --train
```

## ResNet-18

```bash
python3 run.py --model resnet18 --dataset cifar10 --epochs 20 --batch_size 64 --train
```

---

# Testing Commands

## AlexNet

```bash
python3 run.py --model alexnet --dataset cifar10 --test --weights models/alexnet_cifar10_best.pth
```

## GoogLeNet

```bash
python3 run.py --model googlenet --dataset cifar10 --test --weights models/googlenet_cifar10_best.pth
```

## ResNet-18

```bash
python3 run.py --model resnet18 --dataset cifar10 --test --weights models/resnet18_cifar10_best.pth
```

---

# Training Curves

The training and validation curves generated during training are stored in the `plots/` directory.

* `plots/alexnet_cifar10_curves.png`
* `plots/googlenet_cifar10_curves.png`
* `plots/resnet18_cifar10_curves.png`



# Exercise 1 – Question 2

## AlexNet with Local Response Normalization (LRN)

Local Response Normalization (LRN) was incorporated into the AlexNet architecture following the original AlexNet paper. LRN layers were added after the first and second convolutional layers using the parameters described in the paper:

* size = 5
* alpha = 0.0001
* beta = 0.75
* k = 2

The purpose of LRN is to encourage competition among neighboring feature maps and suppress excessively large activations.

## Results

| Model         | Validation Accuracy | Test Accuracy |
| ------------- | ------------------: | ------------: |
| AlexNet       |              62.86% |        63.14% |
| AlexNet + LRN |              51.59% |        51.16% |

## Discussion

The AlexNet model without LRN achieved a test accuracy of 63.14%, whereas the AlexNet model with LRN achieved a test accuracy of 51.16%.

Contrary to the original AlexNet paper, the inclusion of LRN reduced performance on the CIFAR-10 dataset. One possible explanation is that LRN was originally developed for large-scale ImageNet training and older optimization techniques. Modern deep learning frameworks already benefit from improved initialization, optimization methods, and regularization strategies, reducing the usefulness of LRN.

The results suggest that LRN is not beneficial for this CIFAR-10 classification task. This observation is consistent with modern convolutional neural network architectures, which typically use Batch Normalization instead of Local Response Normalization.

## Conclusion

For this experiment, AlexNet without LRN produced significantly better classification performance than AlexNet with LRN. Therefore, LRN was not an effective normalization technique for this particular dataset and training setup.


# Exercise 1 – Question 3

## GoogLeNet with ImageNet-Style Backbone and Auxiliary Classifiers

For this question, I modified the original CIFAR-10 GoogLeNet implementation to more closely follow the original GoogLeNet/Inception-v1 architecture.

The original Q1 GoogLeNet used 32×32 CIFAR-10 images and a simplified convolutional stem. For Q3, I modified the architecture to use 224×224 input images with an ImageNet-style backbone before the first Inception module.

The modified backbone includes:

* 7×7 convolution with stride 2
* max pooling
* local response normalization
* 1×1 convolution
* 3×3 convolution
* local response normalization
* max pooling

I also added two auxiliary classifiers after intermediate Inception blocks. During training, the total loss was calculated as:

```text
Total Loss = Main Loss + 0.3 × Aux Loss 1 + 0.3 × Aux Loss 2
```

During testing, only the final main classifier output was used.

## Results

| Model        | Input Size | Auxiliary Classifiers | Parameters | Best Validation Accuracy | Test Accuracy |
| ------------ | ---------: | --------------------: | ---------: | -----------------------: | ------------: |
| GoogLeNet Q1 |      32×32 |                    No |  6,166,250 |                   85.90% |        85.97% |
| GoogLeNet Q3 |    224×224 |                   Yes | 10,635,134 |                   85.27% |        84.16% |

## Discussion

The modified GoogLeNet with the ImageNet-style backbone and auxiliary classifiers achieved a test accuracy of 84.16%. This was slightly lower than the simplified CIFAR-10 GoogLeNet from Q1, which achieved 85.97%.

Although the Q3 version is closer to the original GoogLeNet paper architecture, it did not improve performance on CIFAR-10. One reason is that CIFAR-10 images are originally only 32×32 pixels, so resizing them to 224×224 does not add new visual information. It only increases computational cost.

The Q3 model also had more parameters and trained much more slowly because of the larger input size and auxiliary classifier branches. Therefore, for CIFAR-10, the smaller 32×32 GoogLeNet implementation was more efficient and slightly more accurate.



# Exercise 1 – Question 4

## Comparison of AlexNet and GoogLeNet

In this experiment, AlexNet and GoogLeNet were trained and evaluated on the CIFAR-10 dataset. The two architectures differ significantly in design philosophy, parameter count, and computational efficiency.

## Experimental Results

| Model     | Parameters | Validation Accuracy | Test Accuracy |
| --------- | ---------: | ------------------: | ------------: |
| AlexNet   | 57,044,810 |              62.86% |        63.14% |
| GoogLeNet |  6,166,250 |              85.90% |        85.97% |

## Architectural Differences

### AlexNet

AlexNet consists of a sequence of convolutional layers followed by large fully connected layers. Most of the model parameters are concentrated in the fully connected layers near the end of the network. The architecture is relatively straightforward and was one of the first deep convolutional neural networks to achieve strong ImageNet performance.

### GoogLeNet

GoogLeNet introduces the Inception module, which processes feature maps through multiple convolution branches simultaneously. This allows the network to capture information at different spatial scales while maintaining a relatively small number of parameters. GoogLeNet also replaces large fully connected layers with global average pooling, significantly reducing parameter count.

## Analysis

Although AlexNet contains approximately 57 million parameters, it achieved only 63.14% test accuracy on CIFAR-10. In contrast, GoogLeNet achieved 85.97% test accuracy while using only about 6.2 million parameters.

This demonstrates that increasing the number of parameters alone does not guarantee better performance. The Inception architecture enables GoogLeNet to learn richer feature representations while remaining parameter-efficient.

GoogLeNet therefore achieved:

* Higher classification accuracy
* Better parameter efficiency
* Improved feature extraction capability
* Lower memory requirements

## Conclusion

GoogLeNet substantially outperformed AlexNet on CIFAR-10 while using nearly nine times fewer parameters. The results demonstrate the effectiveness of the Inception architecture and show that architectural design is often more important than simply increasing model size.

# Exercise 1 – Question 5

## Pretrained AlexNet and GoogLeNet

For this question, pretrained AlexNet and GoogLeNet models were loaded from the torchvision repository. Both models were originally trained on ImageNet, which contains 1000 classes. Since CIFAR-10 contains only 10 classes, the final classification layers were replaced with new 10-class output layers.

A two-stage fine-tuning strategy was used:

1. Freeze the pretrained backbone and train only the new classifier head.
2. Unfreeze all layers and fine-tune the entire network on CIFAR-10.

## Results

| Model     | Training Type | Parameters | Best Validation Accuracy | Test Accuracy |
| --------- | ------------- | ---------: | -----------------------: | ------------: |
| AlexNet   | From scratch  | 57,044,810 |                   62.86% |        63.14% |
| AlexNet   | Pretrained    | 57,044,810 |                   89.76% |        89.71% |
| GoogLeNet | From scratch  |  6,166,250 |                   85.90% |        85.97% |
| GoogLeNet | Pretrained    |  9,960,638 |                   94.27% |        93.71% |

## Discussion

Both pretrained models significantly outperformed their from-scratch versions. AlexNet improved from 63.14% test accuracy to 89.71%, showing that ImageNet-pretrained features greatly improved generalization on CIFAR-10.

GoogLeNet also improved from 85.97% to 93.71%. The improvement was smaller than AlexNet's because the from-scratch GoogLeNet already performed strongly on CIFAR-10. However, the pretrained GoogLeNet still achieved the best performance among the models tested so far.

These results show that pretrained models have strong feature extraction ability. Features learned from ImageNet, such as edges, textures, shapes, and object parts, can transfer effectively to CIFAR-10. The results also suggest that GoogLeNet has better capacity and generalization ability than AlexNet due to its Inception architecture and parameter-efficient design.


# Exercise 1 – Question 6

## Q6(a): Implementation of ResidualBlock and ResNet-18

A ResNet-18 architecture was implemented from scratch using PyTorch. The implementation consists of two main classes:

* `ResidualBlock`
* `ResNet18`

Each residual block contains:

```text
Input x
  ├── Conv → BN → ReLU → Conv → BN
  └── Shortcut Connection
           ↓
        Addition
           ↓
         ReLU
```

When the input and output dimensions differ, a 1×1 convolution is used in the shortcut path to match dimensions.

The ResNet-18 architecture contains:

* Initial convolution layer
* Four residual stages
* Global average pooling
* Fully connected classification layer

---

## Q6(b): Training ResNet-18 from Scratch on CIFAR-10

The implemented ResNet-18 model was trained from scratch on the CIFAR-10 dataset.

### Results

| Model              | Best Validation Accuracy | Test Accuracy |
| ------------------ | -----------------------: | ------------: |
| ResNet18 (Scratch) |                   77.62% |        78.33% |

### Discussion

The ResNet-18 model achieved significantly better performance than AlexNet trained from scratch. The residual connections enabled stable optimization and improved feature learning. Compared to AlexNet, ResNet-18 converged faster and achieved higher classification accuracy on CIFAR-10.

---

## Q6(c): Fine-Tuning a Pretrained ResNet-18

A pretrained ResNet-18 model was loaded from the torchvision repository using ImageNet weights:

```python
resnet_pretrained = torchvision.models.resnet18(
    weights='IMAGENET1K_V1'
)
```

Since CIFAR-10 contains only 10 classes, the original fully connected layer was replaced:

```python
resnet_pretrained.fc = nn.Linear(512, 10)
```

A two-stage fine-tuning strategy was used.

### Stage 1: Freeze Backbone

All pretrained layers were frozen and only the newly initialized classification layer was trained.

```python
for param in resnet_pretrained.parameters():
    param.requires_grad = False

resnet_pretrained.fc.requires_grad_(True)
```

This allows the classifier head to learn CIFAR-10 without disrupting the pretrained ImageNet features.

### Stage 2: Fine-Tune Entire Network

After the classifier stabilized, all layers were unfrozen and the entire network was fine-tuned using a smaller learning rate.

```python
for param in resnet_pretrained.parameters():
    param.requires_grad = True
```

This allows the pretrained features to adapt to CIFAR-10 while preserving useful ImageNet representations.

### Results

| Model                 | Best Validation Accuracy | Test Accuracy |
| --------------------- | -----------------------: | ------------: |
| ResNet18 (Scratch)    |                   77.62% |        78.33% |
| ResNet18 (Pretrained) |                   93.28% |        93.14% |

### Discussion

The pretrained ResNet-18 substantially outperformed the model trained from scratch. The pretrained network already contained useful visual features learned from ImageNet, including edge detectors, texture representations, and object-level patterns. Fine-tuning these pretrained features on CIFAR-10 resulted in a significant improvement in accuracy.

---

## Q6(d): Why Does ResNet Train Deep Networks Successfully?

### Vanishing Gradient Problem

During backpropagation, gradients must pass through many layers of a deep neural network. At each layer, gradients are multiplied by local derivatives.

If these derivatives are consistently smaller than one, gradients become progressively smaller as they travel toward earlier layers.

Example:

```text
1 × 0.5 × 0.5 × 0.5 × 0.5
= 0.0625
```

As network depth increases:

* Early layers receive very small gradients.
* Weight updates become negligible.
* Learning slows down or stops.
* Model performance saturates.

This phenomenon is known as the **vanishing gradient problem**.

### Skip Connections

ResNet introduces shortcut (skip) connections:

```text
Input x
  ├── Conv → BN → ReLU → Conv → BN
  └───────────────────────────────
                 ↓
              Addition
                 ↓
               ReLU
```

Instead of learning a direct mapping (H(x)), the residual block learns:

```text
F(x) = H(x) − x
```

and produces:

```text
Output = F(x) + x
```

### How Skip Connections Help

The shortcut path provides a direct route for gradients during backpropagation.

Benefits include:

* Stronger gradient flow
* Reduced vanishing gradients
* Easier optimization
* Successful training of very deep networks

Because of skip connections, ResNet architectures with 18, 34, 50, 101, and even 152 layers can be trained effectively.

### Conclusion

ResNet successfully trains deep neural networks because skip connections provide a direct pathway for gradient propagation. This alleviates the vanishing gradient problem and enables stable optimization of very deep architectures.

# Exercise 2

## Fine-Tuning a Pretrained ViT-B/16 on CIFAR-10

A pretrained Vision Transformer (ViT-B/16) was loaded from the torchvision repository using ImageNet pretrained weights.

```python
vit_pretrained = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
vit_pretrained.heads = nn.Linear(768, 10)
```

Since ImageNet contains 1000 classes and CIFAR-10 contains only 10 classes, the original classification head was replaced with a new fully connected layer producing 10 outputs.

A two-stage fine-tuning strategy was used.

### Stage 1: Freeze Backbone

All transformer layers were frozen and only the classification head was trained.

```python
for param in vit_pretrained.parameters():
    param.requires_grad = False

vit_pretrained.heads.requires_grad_(True)
```

### Stage 2: Fine-Tune Entire Network

After the classifier head converged, all layers were unfrozen and the entire network was fine-tuned.

```python
for param in vit_pretrained.parameters():
    param.requires_grad = True
```

Since ViT-B/16 expects 224×224 images, CIFAR-10 images were resized before being fed into the model.

---

## Results Table

| Model                                   |   # Params | Test Accuracy | Time/Epoch | Architecture Type      |
| --------------------------------------- | ---------: | ------------: | ---------: | ---------------------- |
| AlexNet + LRN (from scratch)            | 57,044,810 |        51.16% |       ~57s | CNN                    |
| GoogLeNet + 2 Aux Losses (from scratch) | 10,635,134 |        84.16% |      ~190s | CNN + Inception        |
| ResNet-18 (from scratch)                | 11,173,962 |        78.33% |       ~17s | CNN + Skip Connections |
| ResNet-18 (pretrained)                  | 11,181,642 |        93.14% |       ~65s | CNN + Skip Connections |
| ViT-Small (from scratch)                |  1,205,898 |        66.47% |       ~30s | Transformer            |
| ViT-B/16 (pretrained, fine-tuned)       | 86,575,114 |        94.11% |      ~600s | Transformer            |

---

## Discussion

The pretrained ViT-B/16 achieved the highest test accuracy of 94.11% on CIFAR-10. The model benefited from large-scale ImageNet pretraining and the powerful self-attention mechanism of Vision Transformers. Fine-tuning allowed the pretrained representations to adapt effectively to CIFAR-10 while retaining useful visual knowledge learned from ImageNet.

The pretrained ViT-B/16 slightly outperformed the pretrained ResNet-18 (94.11% vs 93.14%). However, the improvement was relatively small considering the significantly larger model size and computational cost of ViT-B/16. This suggests that both pretrained CNNs and Transformers transfer effectively to small datasets, although Transformers generally require more computation and memory.

Among models trained from scratch, GoogLeNet with auxiliary classifiers achieved the highest accuracy (84.16%). The Inception architecture and auxiliary losses helped improve optimization and feature learning while maintaining good parameter efficiency.

CNN-based architectures train efficiently on relatively small datasets because they incorporate useful inductive biases such as locality and weight sharing. Transformer-based architectures are more flexible and can model global relationships through self-attention, but they typically require larger datasets or pretraining to reach their full potential. The results clearly show that pretraining is highly beneficial for both CNNs and Transformers.

### Conclusion

The best overall model was the pretrained ViT-B/16, achieving 94.11% test accuracy. The pretrained ResNet-18 achieved nearly the same performance while requiring significantly fewer parameters and much less computation. These results demonstrate the importance of transfer learning and show that both CNNs and Transformers can achieve excellent performance when pretrained on large-scale datasets.
