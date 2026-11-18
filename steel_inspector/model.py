import hashlib
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .data import CENTER_RANGE, CLASSES, STRIDE, horizontal_flip, preprocess, read_split


def conv(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(),
    )


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(conv(1, 24, 2), conv(24, 40, 2), conv(40, 64, 2), conv(64, 64))
        self.context = nn.Sequential(conv(64, 96, 2), conv(96, 96))
        self.head = nn.Sequential(conv(160, 64), conv(64, 48))
        self.classes = nn.Conv2d(48, len(CLASSES), 1)
        self.regression = nn.Conv2d(48, 4, 1)
        nn.init.constant_(self.classes.bias, -2.19)
        nn.init.constant_(self.regression.bias[2:], -1.1)

    def forward(self, image):
        detail = self.encoder(image)
        context = F.interpolate(self.context(detail), scale_factor=2, mode="nearest")
        features = self.head(torch.cat((detail, context), dim=1))
        return torch.cat((self.classes(features), self.regression(features)), dim=1)


def encode(boxes, labels, width, height, grid):
    classes = np.full((grid, grid), -1, dtype=np.int64)
    target = np.zeros((4, grid, grid), dtype=np.float32)
    normalized = boxes / np.array([width, height, width, height], dtype=np.float32)
    sizes = normalized[:, 2:] - normalized[:, :2]
    centers = (normalized[:, :2] + normalized[:, 2:]) * (grid / 2)
    for i in np.argsort(sizes.prod(axis=1), kind="stable"):
        cx, cy = centers[i]
        for y in range(max(0, math.floor(cy - 1)), min(grid, math.ceil(cy + 1))):
            for x in range(max(0, math.floor(cx - 1)), min(grid, math.ceil(cx + 1))):
                dx, dy = cx - x - 0.5, cy - y - 0.5
                if classes[y, x] < 0 and max(abs(dx), abs(dy)) < CENTER_RANGE:
                    classes[y, x] = labels[i]
                    target[:, y, x] = [dx / CENTER_RANGE, dy / CENTER_RANGE, *sizes[i]]
    return classes, target


class TrainingImages(Dataset):
    def __init__(self, manifest, size, augment):
        root, rows = read_split(manifest, "train")
        self.samples = []
        self.augment = augment
        for row in rows:
            with Image.open(root / row["file"]) as image:
                image = image.convert("L")
            boxes = np.asarray(row["boxes"], dtype=np.float32)
            variants = [(image, boxes)]
            if augment:
                variants.append(horizontal_flip(image, boxes))
            self.samples.append([
                (preprocess(pixels, size)[0], *encode(coords, row["labels"], *pixels.size, size // STRIDE))
                for pixels, coords in variants
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        variant = int(self.augment and random.random() < 0.5)
        return self.samples[index][variant]


def detection_loss(outputs, labels, coordinates):
    logits, boxes = outputs[:, :len(CLASSES)], outputs[:, len(CLASSES):]
    positive = labels >= 0
    class_ids = torch.arange(len(CLASSES), device=labels.device)[None, :, None, None]
    target = (class_ids == labels[:, None]).to(logits.dtype)
    prob = logits.sigmoid()
    focal_weight = torch.where(target.bool(), 0.25 * (1 - prob).square(), 0.75 * prob.square())
    classification = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    classification = (classification * focal_weight).sum() / positive.sum().clamp_min(1)

    raw = boxes.permute(0, 2, 3, 1)[positive]
    expected = coordinates.permute(0, 2, 3, 1)[positive]
    actual = torch.cat((raw[:, :2].tanh(), raw[:, 2:].sigmoid()), dim=1)
    difference = F.smooth_l1_loss(actual, expected, beta=0.10)
    grid = outputs.shape[-1]
    actual_center = actual[:, :2] * (CENTER_RANGE / grid)
    expected_center = expected[:, :2] * (CENTER_RANGE / grid)
    actual_min = actual_center - actual[:, 2:] / 2
    actual_max = actual_center + actual[:, 2:] / 2
    expected_min = expected_center - expected[:, 2:] / 2
    expected_max = expected_center + expected[:, 2:] / 2
    overlap = (torch.minimum(actual_max, expected_max) - torch.maximum(actual_min, expected_min)).clamp_min(0).prod(dim=1)
    union = actual[:, 2:].prod(dim=1) + expected[:, 2:].prod(dim=1) - overlap
    enclosing = (torch.maximum(actual_max, expected_max) - torch.minimum(actual_min, expected_min)).prod(dim=1)
    giou = overlap / union.clamp_min(1e-8) - (enclosing - union) / enclosing.clamp_min(1e-8)
    return classification + 2 * (1 - giou).mean() + 2 * difference


def train(manifest, output, *, epochs=30, batch_size=32, augment=False, size=192, seed=42, device="cpu"):
    if min(epochs, batch_size) < 1 or size < 32 or size % 16:
        raise ValueError("Positive epochs/batch size and an image size divisible by 16 (minimum 32) are required")
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dataset = TrainingImages(manifest, size, augment)
    batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=device == "cuda")
    model = Detector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)
    history = []
    for epoch in range(epochs):
        loss_sum = 0.0
        for images, labels, coordinates in batches:
            images, labels, coordinates = (value.to(device, non_blocking=True) for value in (images, labels, coordinates))
            optimizer.zero_grad(set_to_none=True)
            loss = detection_loss(model(images), labels, coordinates)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item() * len(images)
        scheduler.step()
        history.append(loss_sum / len(dataset))
        print(f"epoch {epoch + 1}/{epochs} loss={history[-1]:.4f}", flush=True)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": 1,
        "classes": list(CLASSES),
        "size": size,
        "weights": model.cpu().state_dict(),
        "training": {
            "augment": augment,
            "seed": seed,
            "batch_size": batch_size,
            "device": device,
            "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
            "loss_by_epoch": history,
        },
    }, output)
    return history


class HorizontalEnsemble(nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, image):
        direct = self.detector(image)
        reflected = self.detector(image.flip((-1,))).flip((-1,))
        # The horizontal center offset changes sign when mapped to the original image.
        offset = len(CLASSES)
        aligned = torch.cat((reflected[:, :offset], -reflected[:, offset:offset + 1], reflected[:, offset + 1:]), dim=1)
        return (direct + aligned) / 2


def restore(checkpoint, *, flip_ensemble=False):
    checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["version"] != 1 or checkpoint["classes"] != list(CLASSES):
        raise ValueError("Unsupported checkpoint format")
    model = Detector()
    model.load_state_dict(checkpoint["weights"])
    if flip_ensemble:
        model = HorizontalEnsemble(model)
    return model.eval(), checkpoint["size"]


def export_model(checkpoint, output, *, flip_ensemble=False):
    import onnx

    model, size = restore(checkpoint, flip_ensemble=flip_ensemble)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, torch.zeros(1, 1, size, size), output,
        input_names=["image"], output_names=["predictions"], opset_version=17, dynamo=False,
    )
    onnx.checker.check_model(output)
