from __future__ import division
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import os
import json
import math
import argparse
import time
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Subset
from torchvision.datasets import CocoDetection
from typing import Callable, Optional, Tuple
import albumentations as A
import darknet
from util import *

# ─── Constants ───────────────────────────────────────────────
img_size    = 608
NUM_CLASSES = 80
NUM_ANCHORS = 3
BATCH_SIZE  = 2

ANCHORS = [
    [[12, 16], [19, 36], [40, 28]],
    [[36, 75], [76, 55], [72, 146]],
    [[142, 110], [192, 243], [459, 401]]
]
STRIDES = [8, 16, 32]

path2data = '/home/jupyter-st125985/fiftyone/coco-2017/validation/data/val2017'
path2json = '/home/jupyter-st125985/fiftyone/coco-2017/raw/annotations/instances_val2017.json'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load COCO categories
with open('coco_cats.json') as js:
    data = json.load(js)["categories"]
cats_dict = {str(data[i]['id']): i for i in range(80)}


# ─── Helper Functions ─────────────────────────────────────────
def iou_xywh_numpy(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]
    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                              boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                              boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    return 1.0 * inter_area / (union_area + 1e-6)


def CIOU_xywh_torch(boxes1, boxes2):
    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    inter_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    inter_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = torch.max(inter_right_down - inter_left_up, torch.zeros_like(inter_right_down))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = 1.0 * inter_area / (union_area + 1e-6)
    outer_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    outer_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    outer = torch.max(outer_right_down - outer_left_up, torch.zeros_like(inter_right_down))
    outer_diagonal_line = torch.pow(outer[..., 0], 2) + torch.pow(outer[..., 1], 2)
    boxes1_center = (boxes1[..., :2] + boxes1[..., 2:]) * 0.5
    boxes2_center = (boxes2[..., :2] + boxes2[..., 2:]) * 0.5
    center_dis = torch.pow(boxes1_center[..., 0] - boxes2_center[..., 0], 2) + \
                 torch.pow(boxes1_center[..., 1] - boxes2_center[..., 1], 2)
    boxes1_size = torch.max(boxes1[..., 2:] - boxes1[..., :2], torch.zeros_like(inter_right_down))
    boxes2_size = torch.max(boxes2[..., 2:] - boxes2[..., :2], torch.zeros_like(inter_right_down))
    v = (4 / (math.pi ** 2)) * torch.pow(
        torch.atan(boxes1_size[..., 0] / torch.clamp(boxes1_size[..., 1], min=1e-6)) -
        torch.atan(boxes2_size[..., 0] / torch.clamp(boxes2_size[..., 1], min=1e-6)), 2)
    alpha = v / (1 - ious + v + 1e-6)
    return ious - (center_dis / (outer_diagonal_line + 1e-6) + alpha * v)

# ─── Dataset ──────────────────────────────────────────────────
class CustomCoco(CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = np.array(img)
        category_ids = [obj['category_id'] for obj in target]
        bboxes = [obj['bbox'] for obj in target]
        if self.transform is not None:
            transformed = self.transform(image=img, bboxes=bboxes, category_ids=category_ids)
            img = transformed['image']
            bboxes = torch.Tensor(transformed['bboxes'])
            cat_ids = torch.Tensor(transformed['category_ids'])
            labels, bboxes = self.__create_label(bboxes, cat_ids.type(torch.IntTensor))
        return img, labels, bboxes

    def __len__(self):
        return len(self.ids)

    def __create_label(self, bboxes, class_inds):
        bboxes = np.array(bboxes)
        class_inds = np.array(class_inds)
        strides = np.array(STRIDES)
        train_output_size = img_size / strides
        anchors_per_scale = NUM_ANCHORS
        label = [
            np.zeros((int(train_output_size[i]), int(train_output_size[i]),
                      anchors_per_scale, 5 + NUM_CLASSES))
            for i in range(3)
        ]
        bboxes_xywh = [np.zeros((150, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for i in range(len(bboxes)):
            bbox_coor = bboxes[i][:4]
            bbox_class_ind = cats_dict[str(class_inds[i])]
            one_hot = np.zeros(NUM_CLASSES, dtype=np.float32)
            one_hot[bbox_class_ind] = 1.0
            bbox_xywh = np.concatenate([
                (0.5 * bbox_coor[2:] + bbox_coor[:2]),
                bbox_coor[2:]
            ], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]
            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((anchors_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = ANCHORS[i]
                iou_scale = iou_xywh_numpy(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3
                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = one_hot
                    bbox_ind = int(bbox_count[i] % 150)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1
                    exist_positive = True
            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / anchors_per_scale)
                best_anchor = int(best_anchor_ind % anchors_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
           
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = one_hot
                bbox_ind = int(bbox_count[best_detect] % 150)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        flatten_size_s = int(train_output_size[2]) * int(train_output_size[2]) * anchors_per_scale
        flatten_size_m = int(train_output_size[1]) * int(train_output_size[1]) * anchors_per_scale
        flatten_size_l = int(train_output_size[0]) * int(train_output_size[0]) * anchors_per_scale
        label_s = torch.Tensor(label[2]).view(1, flatten_size_s, 5 + NUM_CLASSES).squeeze(0)
        label_m = torch.Tensor(label[1]).view(1, flatten_size_m, 5 + NUM_CLASSES).squeeze(0)
        label_l = torch.Tensor(label[0]).view(1, flatten_size_l, 5 + NUM_CLASSES).squeeze(0)
        labels = torch.cat([label_l, label_m, label_s], 0)
        bboxes = torch.cat([torch.Tensor(bboxes_xywh[0]),
                            torch.Tensor(bboxes_xywh[1]),
                            torch.Tensor(bboxes_xywh[2])], 0)
        return labels, bboxes

# ─── Training Function ────────────────────────────────────────
def train_yolo(model, optimizer, dataloader, device, img_size, n_epoch, use_ciou=False):
    model.train()
    loss_history = []

    for epoch in range(n_epoch):
        running_loss = 0.0
        valid_batches = 0
        start_time = time.time()

        for batch_idx, (inputs, labels, bboxes) in enumerate(dataloader):
            inputs = torch.from_numpy(np.array(inputs)).squeeze(1).permute(0, 3, 1, 2).float()
            inputs = inputs / 255.0
            inputs = inputs.to(device)

            labels = torch.stack(labels).to(device)

            optimizer.zero_grad()

            outputs = model(inputs, torch.cuda.is_available())

            if not torch.isfinite(outputs).all():
                print(f"Skipping batch {batch_idx}: model output has NaN/Inf")
                continue

            pred_xywh = outputs[..., 0:4] / img_size
            pred_conf = outputs[..., 4:5]
            pred_cls = outputs[..., 5:]

            label_xywh = labels[..., :4] / img_size
            label_obj_mask = labels[..., 4:5]
            label_noobj = 1.0 - label_obj_mask
            label_cls = labels[..., 5:]

            bce = nn.BCELoss(reduction="none")

            pred_conf = torch.clamp(pred_conf, 1e-6, 1.0 - 1e-6)
            pred_cls = torch.clamp(pred_cls, 1e-6, 1.0 - 1e-6)

            if use_ciou:
                ciou = CIOU_xywh_torch(pred_xywh, label_xywh).unsqueeze(-1)
                ciou = torch.clamp(ciou, min=-1.0, max=1.0)
                loss_box = torch.sum(label_obj_mask * (1.0 - ciou))
            else:
                mse = nn.MSELoss(reduction="none")
                loss_box = torch.sum(label_obj_mask * mse(pred_xywh, label_xywh))

            loss_conf = torch.sum(
                label_obj_mask * bce(pred_conf, label_obj_mask)
                + 0.05 * label_noobj * bce(pred_conf, label_obj_mask)
            )

            loss_cls = torch.sum(label_obj_mask * bce(pred_cls, label_cls))

            loss = loss_box + loss_conf + loss_cls

            if not torch.isfinite(loss):
                print(f"Skipping NaN/Inf loss at epoch {epoch+1}, batch {batch_idx}")
                print("loss_box :", loss_box)
                print("loss_conf:", loss_conf)
                print("loss_cls :", loss_cls)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += loss.item()
            valid_batches += 1

            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

        epoch_loss = running_loss / max(valid_batches, 1)
        epoch_time = time.time() - start_time
        loss_history.append(epoch_loss)

        print(
            f">>> Epoch {epoch+1}/{n_epoch}  "
            f"Loss: {epoch_loss:.4f}  "
            f"Valid batches: {valid_batches}/{len(dataloader)}  "
            f"Time: {epoch_time:.1f}s"
        )

    return loss_history

# ─── mAP Evaluation Function ──────────────────────────────────
# ─── mAP Evaluation Function ──────────────────────────────────
def evaluate_map(model, device, img_size, conf=0.5, nms=0.4):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    model.eval()
    coco_gt = COCO(path2json)
    results = []
    img_ids = []
    eval_dataset = Subset(
        CustomCoco(root=path2data, annFile=path2json, transform=A.Compose([
            A.Resize(img_size, img_size)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))),
        list(range(0, 500))
    )
    with torch.no_grad():
        for idx in range(len(eval_dataset)):
            img, labels, bboxes = eval_dataset[idx]

            # get real image id
            real_idx = eval_dataset.indices[idx]
            img_id = eval_dataset.dataset.ids[real_idx]
            img_ids.append(img_id)

            inputs = torch.from_numpy(np.array(img)).float().div(255.0)
            inputs = inputs.permute(2,0,1).unsqueeze(0).to(device)

            preds = model(inputs, torch.cuda.is_available())
            dets = write_results(preds, conf, NUM_CLASSES, nms_conf=nms)

            # debug first image
            if idx == 0:
                print(f"img_id: {img_id}, dets type: {type(dets)}")
                if type(dets) != int:
                    print(f"First det: {dets[0]}")

            if type(dets) == int:
                continue

            orig = coco_gt.loadImgs(img_id)[0]
            ow, oh = orig['width'], orig['height']

            for det in dets:
                x1 = float(det[1]) * ow / img_size
                y1 = float(det[2]) * oh / img_size
                x2 = float(det[3]) * ow / img_size
                y2 = float(det[4]) * oh / img_size
                w = x2 - x1
                h = y2 - y1
                if w <= 5 or h <= 5:
                    continue
                score = float(det[5])
                cls = int(det[7])
                results.append({
                    "image_id": img_id,
                    "category_id": int(data[cls]['id']),
                    "bbox": [x1, y1, w, h],
                    "score": score
                })

            if idx % 50 == 0:
                print(f"Evaluated {idx}/{len(eval_dataset)} images, detections so far: {len(results)}")

    if len(results) == 0:
        print("No detections found!")
        return 0.0
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0]
# ─── Inference Function ───────────────────────────────────────
def infer(model_name, weights, image_path):
    cfg = f"cfg/{model_name}.cfg"
    model = darknet.Darknet(cfg)
    model.load_weights(weights)
    model.net_info["height"] = str(img_size)
    model.to(device)
    model.eval()

    classes = load_classes("data/coco.names")
    colors = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255]]

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_tensor = torch.from_numpy(img_resized).float().div(255.0)
    img_tensor = img_tensor.permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor, torch.cuda.is_available())
    output = write_results(preds, 0.5, NUM_CLASSES, nms_conf=0.4)

    if type(output) == int:
        print("No detections!")
        return

    orig_h, orig_w = img.shape[:2]
    scale_x = orig_w / img_size
    scale_y = orig_h / img_size

    for det in output:
        x1 = int(det[1] * scale_x)
        y1 = int(det[2] * scale_y)
        x2 = int(det[3] * scale_x)
        y2 = int(det[4] * scale_y)
        cls = int(det[7])
        label = classes[cls]
        color = colors[cls % len(colors)]
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        print(f"Detected: {label} ({float(det[5]):.2f})")

    out_path = "detection_result.jpg"
    cv2.imwrite(out_path, img)
    print(f"Result saved to {out_path}")


# ─── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="yolov4")
    parser.add_argument("--weights",  default="yolov4.weights")
    parser.add_argument("--image",    default="dog-cycle-car.png")
    parser.add_argument("--dataset",  default="coco")
    parser.add_argument("--epochs",   type=int, default=5)
    parser.add_argument("--infer",    action="store_true")
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    if args.infer:
        print(f"Running inference with {args.model}...")
        infer(args.model, args.weights, args.image)

    elif args.train:
        print(f"Training {args.model} for {args.epochs} epochs...")

        def collate_fn(batch):
            return tuple(zip(*batch))

        train_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

        train_dataset = Subset(
            CustomCoco(root=path2data, annFile=path2json, transform=train_transform),
            list(range(0, 2000))
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4, collate_fn=collate_fn
        )

        # Train with MSE loss
        print("\n--- Training with MSE loss ---")
        model = darknet.Darknet(f"cfg/{args.model}.cfg")
        model.load_weights(args.weights)
        model.net_info["height"] = str(img_size)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        loss_mse = train_yolo(model, optimizer, train_dataloader, device, img_size, args.epochs, use_ciou=False)
        torch.save(model.state_dict(), "yolov4_mse.pth")
        print("MSE model saved!")

        # Train with CIoU loss
        print("\n--- Training with CIoU loss ---")
        model2 = darknet.Darknet(f"cfg/{args.model}.cfg")
        model2.load_weights(args.weights)
        model2.net_info["height"] = str(img_size)
        model2.to(device)
        optimizer2 = optim.Adam(model2.parameters(), lr=0.00001)
        loss_ciou = train_yolo(model2, optimizer2, train_dataloader, device, img_size, args.epochs, use_ciou=True)
        torch.save(model2.state_dict(), "yolov4_ciou.pth")
        print("CIoU model saved!")

        print("\n=== Loss Comparison ===")
        print(f"MSE  final loss: {loss_mse[-1]:.4f}")
        print(f"CIoU final loss: {loss_ciou[-1]:.4f}")

    elif args.evaluate:
        print(f"Evaluating {args.model}...")
        model = darknet.Darknet(f"cfg/{args.model}.cfg")
        model.net_info["height"] = str(img_size)
        if args.weights.endswith(".pth"):
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            model.load_weights(args.weights)
        model.to(device)
        model.eval()
        map_val = evaluate_map(model, device, img_size, conf=0.25, nms=0.5)
        print(f"\nmAP: {map_val:.4f}")