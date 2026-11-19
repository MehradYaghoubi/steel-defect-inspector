import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils import loss

PROJECT = Path(__file__).resolve().parent
RUNS = PROJECT / "runs"
SLF = RUNS / "slf-reference/SLF-YOLO"


def fimetal_iou(predicted, target, *, xywh=False, CIoU=True):
    assert not xywh and CIoU
    ratio, lower, upper = 1.20, 0.0, 0.95
    lower_left, upper_right = predicted.float().chunk(2, dim=-1)
    target_low, target_high = target.float().chunk(2, dim=-1)
    grow = (ratio - 1.0) / 2.0
    extra = (upper_right - lower_left) * grow
    target_extra = (target_high - target_low) * grow
    overlap = (
        torch.minimum(upper_right + extra, target_high + target_extra)
        - torch.maximum(lower_left - extra, target_low - target_extra)
    ).clamp(min=0).prod(dim=-1, keepdim=True)
    area = ((upper_right - lower_left) * ratio).prod(dim=-1, keepdim=True)
    target_area = ((target_high - target_low) * ratio).prod(dim=-1, keepdim=True)
    inner_iou = overlap / (area + target_area - overlap).clamp_min(1e-7)
    return ((inner_iou - lower) / (upper - lower)).clamp(0.0, 1.0)


def check_fimetal():
    pred = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [5.0, 0.0, 15.0, 10.0],
        [20.0, 0.0, 30.0, 10.0],
    ], requires_grad=True)
    target = torch.tensor([[0.0, 0.0, 10.0, 10.0]]).expand_as(pred)
    values = fimetal_iou(pred, target)
    expected = torch.tensor([[1.0], [(7 / 17) / 0.95], [0.0]])
    torch.testing.assert_close(values, expected)
    (1 - values).sum().backward()
    assert torch.isfinite(pred.grad).all() and pred.grad[1].abs().sum() > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--box-loss", required=True, choices=("ciou", "fimetal"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    assert Path(loss.__file__).resolve().is_relative_to(SLF)
    assert hashlib.sha256((PROJECT / "data/split.json").read_bytes()).hexdigest() == (
        "3e278f74bf136cbcd149a0281ed8e3b661df24f8ad9c0ec3c1d1819ebd4dcead"
    )
    if args.device == "cpu":
        torch.set_num_threads(2)
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    check_fimetal()
    if args.box_loss == "fimetal":
        loss.bbox_iou = fimetal_iou

    architecture = yaml.safe_load((SLF / "new_cfg/yolov8-SlimAsfNeck-CGLU.yaml").read_bytes())
    architecture["nc"] = 6
    config = RUNS / "yolov8s-slf-matched.yaml"
    if config.exists():
        if yaml.safe_load(config.read_bytes()) != architecture:
            raise ValueError("Existing SLF configuration differs from the six-class release architecture")
    else:
        config.write_text(yaml.safe_dump(architecture, sort_keys=False), encoding="utf-8")
    model = YOLO(str(config))
    assert sum(parameter.numel() for parameter in model.model.parameters()) == 9666942
    model.train(
        data=str(RUNS / "yolo-neudet/dataset.yaml"),
        project=str(RUNS / "slf-matched"), name=args.name,
        device=args.device, seed=42, pretrained=False, optimizer="SGD",
        imgsz=args.imgsz, batch=args.batch, epochs=args.epochs, fraction=args.fraction,
        lr0=0.01, momentum=0.937, weight_decay=0.0005, close_mosaic=10,
        patience=0, workers=0, cache=False, plots=False, amp=False,
    )
