# A2-01: Object Detection — YOLOv4

## Overview
This assignment extends the YOLOv3 Darknet implementation to support YOLOv4, trains it on the COCO dataset, and compares MSE loss vs CIoU loss.

---

## Exercise 1.1 — Porting YOLOv3 to YOLOv4

The following changes were made to `darknet.py` to support YOLOv4:

### Changes Made

| Change | Description |
|---|---|
| Mish activation | Added `Mish` class using `x * tanh(softplus(x))` |
| Maxpool support | Added `[maxpool]` block in `create_modules()` and `forward()` |
| Route layer fix | Generalized to concatenate any number of layers (YOLOv4 uses 4) |
| Input size | Changed from 416×416 to 608×608 |
| Color order | Changed from BGR to RGB |

---

## Exercise 1.2 — Training YOLOv4 on COCO

### Dataset
- **Dataset**: COCO 2017 validation set
- **Training images**: 2000 images
- **Evaluation images**: 500 images
- **Image size**: 608×608

### a) Pretrained Weights
Loaded pretrained YOLOv4 weights (CSPDarknet53 backbone) from the official Darknet release:
```python
model = darknet.Darknet("cfg/yolov4.cfg")
model.load_weights("yolov4.weights")
```

### b) train_yolo() Function
The training function implements:
- **Data augmentation**: horizontal flip, random brightness/contrast
- **Anchor assignment**: IoU threshold 0.3 to match ground truth boxes to anchors
- **Loss functions**: MSE or CIoU for bbox + BCE for objectness + BCE for class
- **Gradient clipping**: max_norm=10.0 to prevent exploding gradients
- **NaN detection**: skips batches with invalid outputs

### c) Training Loss (5 epochs)

**MSE Loss Training:**
```
Epoch 1: 192.30
Epoch 2: 127.28
Epoch 3:  98.09
Epoch 4:  84.96
Epoch 5:  75.71  ← decreasing ✅
```

**CIoU Loss Training:**
```
Epoch 1: 219.78
Epoch 2: 155.05
Epoch 3: 126.16
Epoch 4: 112.27
Epoch 5: 104.73  ← decreasing ✅
```

### d) mAP Results

| Model | Dataset | mAP@[0.5:0.95] | Notes |
|---|---|---|---|
| YOLOv3 (pretrained) | COCO val | — | inference only |
| YOLOv4 (pretrained) | COCO val | 0.504 | reference |
| YOLOv4 (MSE loss) | COCO val | 0.0744 | 10 epochs, 2000 images |
| YOLOv4 (CIoU loss) | COCO val | 0.1811 | 10 epochs, 2000 images |

### e) Loss Comparison

| Loss | mAP |
|---|---|
| MSE / IoU loss | 0.0744 |
| CIoU loss | **0.1811** |

CIoU loss is **2.4x better** than MSE loss!

---

## Exercise 1.3 — Why is YOLOv3 Faster than Faster R-CNN?

Faster R-CNN is a **two-stage detector**. In the first stage, a Region Proposal Network (RPN) generates ~300 candidate bounding boxes. In the second stage, each proposal is passed through ROI Pooling and a classification head separately. This two-stage process adds significant computational overhead even though the backbone is shared.

YOLOv3 is a **single-shot detector**. It makes all bounding box and class predictions in a single forward pass over the entire image. The output tensor directly encodes predictions at three grid scales (19×19, 38×38, 76×76) with 3 anchors each — no separate proposal stage, no ROI pooling, no second-stage classifier.

This single-pass architecture is why YOLOv3 achieves ~30fps while Faster R-CNN runs at ~5fps on the same hardware.

---

## Commands Used

```bash
# Inference with pretrained YOLOv3 weights
python3 run.py --model yolov3 --weights yolov3.weights --image dog-cycle-car.png --infer

# Train YOLOv4 on COCO (MSE and CIoU loss)
python3 run.py --model yolov4 --weights yolov4.weights --dataset coco --epochs 10 --train

# Evaluate pretrained YOLOv4
python3 run.py --model yolov4 --weights yolov4.weights --evaluate

# Evaluate MSE trained model
python3 run.py --model yolov4 --weights yolov4_mse.pth --evaluate

# Evaluate CIoU trained model
python3 run.py --model yolov4 --weights yolov4_ciou.pth --evaluate
```

---

## Discussion

**Effect of CIoU vs standard loss:**
CIoU loss achieved mAP of 0.1811 compared to 0.0744 for MSE loss — a 2.4x improvement. This is because CIoU optimizes three components simultaneously: the overlap area between predicted and ground truth boxes, the distance between their centers, and the similarity of their aspect ratios. MSE only minimizes raw coordinate differences without considering box overlap quality.

**Challenges encountered:**
Training on COCO with limited data (2000 images) and epochs (10) resulted in lower mAP compared to the pretrained model (0.504). The main challenges were gradient explosion with MSE loss at higher learning rates, requiring gradient clipping and a very low learning rate (1e-5). NaN losses appeared occasionally due to extreme bounding box predictions, which were handled by skipping invalid batches. With more training data and epochs, the fine-tuned model mAP would approach the pretrained reference.

---

## Repository Structure

```
A2/
├── darknet.py              ← modified for YOLOv4 (Mish, maxpool, route)
├── util.py                 ← utility functions
├── run.py                  ← training/inference/evaluation script
├── cfg/
│   ├── yolov3.cfg
│   └── yolov4.cfg
├── data/
│   └── coco.names
├── logs/
│   ├── infer_yolov3.txt
│   ├── eval_yolov4_pretrained.txt
│   ├── eval_yolov4_mse.txt
│   └── eval_yolov4_ciou.txt
└── README.md
```