from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import ImageDraw, ImageFont

from .data import CENTER_RANGE, CLASSES, STRIDE, preprocess

NMS_IOU = 0.5
MAX_DETECTIONS = 100
COLORS = ("#ffd166", "#52dfbb", "#94baff", "#ff98af", "#e4b6ff", "#ffab78")


@dataclass(frozen=True)
class Runtime:
    size: int
    run: Callable[[np.ndarray], np.ndarray]


def load_torch(checkpoint, *, device="cpu", threads=1):
    import torch

    from .model import restore

    if device == "cpu":
        torch.set_num_threads(threads)
    model, size = restore(checkpoint)
    model.to(device)

    @torch.inference_mode()
    def run(batch):
        return model(torch.from_numpy(batch).to(device)).cpu().numpy()

    return Runtime(size, run)


def load_onnx(path, *, threads=1):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (
        len(inputs) != 1 or len(outputs) != 1
        or inputs[0].name != "image" or outputs[0].name != "predictions"
        or len(inputs[0].shape) != 4
    ):
        raise ValueError("Unsupported ONNX detector input or output")
    batch, channels, height, width = inputs[0].shape
    if (batch, channels) != (1, 1) or not isinstance(height, int) or height < 32 or height != width or height % 16:
        raise ValueError("Unsupported ONNX detector image size")
    if outputs[0].shape != [1, len(CLASSES) + 4, height // STRIDE, height // STRIDE]:
        raise ValueError("Unsupported ONNX prediction shape")
    return Runtime(height, lambda pixels: session.run(["predictions"], {"image": pixels})[0])


def iou(box, boxes):
    intersection = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])).prod(axis=1)
    area = (box[2:] - box[:2]).prod()
    other_area = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    return intersection / np.maximum(area + other_area - intersection, 1e-8)


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -80, 80)))


def decode(output, original_size, *, threshold):
    if not 0 < threshold <= 1:
        raise ValueError("Threshold must be in (0, 1]")
    values = output[0]
    scores = sigmoid(values[:len(CLASSES)])
    labels, ys, xs = np.where(scores >= threshold)
    scores = scores[labels, ys, xs]
    predicted = values[len(CLASSES):, ys, xs].T
    width, height = original_size
    grid_h, grid_w = values.shape[1:]
    centers = (np.column_stack((xs, ys)) + 0.5 + CENTER_RANGE * np.tanh(predicted[:, :2]))
    centers *= [width / grid_w, height / grid_h]
    half_sizes = sigmoid(predicted[:, 2:]) * np.array([width / 2, height / 2])
    boxes = np.column_stack((centers - half_sizes, centers + half_sizes))
    boxes = np.clip(boxes, 0, [width, height, width, height])
    valid = ((boxes[:, 2:] - boxes[:, :2]) > 0).all(axis=1)
    order = np.argsort(-scores, kind="stable")
    order = order[valid[order]]
    keep = []
    while order.size and len(keep) < MAX_DETECTIONS:
        index = order[0]
        keep.append(index)
        remaining = order[1:]
        same_class = labels[remaining] == labels[index]
        suppress = np.zeros(len(remaining), dtype=bool)
        suppress[same_class] = iou(boxes[index], boxes[remaining[same_class]]) > NMS_IOU
        order = remaining[~suppress]
    return boxes[keep], scores[keep], labels[keep]


def predict(runtime, image, *, threshold):
    return decode(runtime.run(preprocess(image, runtime.size)), image.size, threshold=threshold)


def annotate(image, boxes, labels, captions):
    scale = min(3.0, 900 / max(image.size))
    canvas = image.convert("RGB").resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    if len(boxes) == 0:
        return canvas
    coordinates = np.asarray(boxes) * ([canvas.width / image.width, canvas.height / image.height] * 2)
    items = list(zip(coordinates, labels, captions))
    for box, label, caption in reversed(items):
        x1, y1, x2, y2 = box
        color = COLORS[label] if label >= 0 else "#f3f5fb"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        _, _, width, height = draw.textbbox((0, 0), caption, font=font)
        left = max(0, min(x1, canvas.width - width - 8))
        top = max(0, y1 - height - 8)
        draw.rectangle((left, top, left + width + 8, top + height + 8), fill=color)
        draw.text((left + 4, top + 2), caption, fill="#121925", font=font)
    return canvas
