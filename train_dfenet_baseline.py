"""Retrain the released DFE-Net architecture on a project split manifest."""

import argparse
import ast
import hashlib
import subprocess
import sys
from pathlib import Path

from benchmark_yolo import prepare
from steel_inspector.data import write_json


REFERENCE_REVISION = "936067614ef39325c91b85f77995f648305e08e9"


def build_model(reference, architecture):
    import torch
    from ultralytics import YOLO
    from ultralytics.nn import tasks

    if not Path(tasks.__file__).resolve().is_relative_to(reference):
        raise RuntimeError("Expected the Ultralytics fork bundled with DFE-Net")

    # The parser references optional architecture names even for the DFE-Net YAML.
    original_imports = subprocess.check_output(
        ["git", "-C", str(reference), "show", "HEAD:ultralytics/nn/extra_modules/__init__.py"],
        encoding="utf-8",
    )

    def unavailable(self, *args, **kwargs):
        raise RuntimeError(f"Optional dependency unavailable for {type(self).__name__}")

    for node in ast.parse(original_imports).body:
        if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith(
            ("mamba.legacy_mamba_", "head.DyHeadDCN", "head.DyHeadPrune")
        ):
            continue
        for alias in node.names:
            name = alias.asname or alias.name
            if not hasattr(tasks, name):
                setattr(tasks, name, type(name, (torch.nn.Module,), {"__init__": unavailable}))

    model = YOLO(str(architecture), task="detect")
    if sum(parameter.numel() for parameter in model.model.parameters()) != 2_246_720:
        raise ValueError("Unexpected DFE-Net parameter count")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Pinned DFE-Net checkout")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--fraction", type=float, default=1.0)
    args = parser.parse_args()
    if min(args.epochs, args.imgsz, args.batch) <= 0 or not 0 < args.fraction <= 1:
        parser.error("Epochs, image size and batch must be positive, with fraction in (0, 1]")

    reference = args.reference.resolve(strict=True)
    revision = subprocess.check_output(["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip()
    if revision != REFERENCE_REVISION:
        parser.error(f"Expected DFE-Net revision {REFERENCE_REVISION}, got {revision}")

    import torch
    import yaml

    sys.path.insert(0, str(reference))
    if args.device == "cpu":
        torch.set_num_threads(2)
    architecture = reference / "configs/yolo11n-dfe-net.yaml"
    recipe = yaml.safe_load((reference / "configs/train_args_dfe-net.yaml").read_text(encoding="utf-8"))
    recipe.pop("mode", None)
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    model = build_model(reference, architecture)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prepare(args.manifest, output / "dataset", "raw")
    recipe.update(
        data=str(output / "dataset/dataset.yaml"), project=str(output), name="dfenet", seed=42,
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, fraction=args.fraction,
        device=args.device, pretrained=False, workers=0, plots=False, amp=args.device != "cpu",
    )
    write_json(output / "recipe.json", dict(
        reference_revision=revision, manifest_sha256=manifest_sha256,
        architecture_sha256=hashlib.sha256(architecture.read_bytes()).hexdigest(), arguments=recipe,
    ))
    model.train(**recipe)


if __name__ == "__main__":
    main()
