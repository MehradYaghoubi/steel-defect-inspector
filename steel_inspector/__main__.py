import argparse
from pathlib import Path

from .data import prepare


def path_argument(parser, name, **kwargs):
    parser.add_argument(f"--{name}", type=Path, required=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Train and deploy a six-class steel defect detector")
    actions = parser.add_subparsers(dest="action", required=True)

    prep = actions.add_parser("prepare", help="Generate a reproducible train/validation/test split from NEU-DET VOC XML")
    path_argument(prep, "source")
    path_argument(prep, "manifest")
    prep.add_argument("--seed", type=int, default=42)

    fit = actions.add_parser("train", help="Train the PyTorch detector")
    path_argument(fit, "manifest")
    path_argument(fit, "output")
    fit.add_argument("--epochs", type=int, default=30)
    fit.add_argument("--batch-size", type=int, default=32)
    fit.add_argument("--size", type=int, default=192)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    fit.add_argument("--augment", action="store_true", help="Train with random horizontal flips")

    evaluate = actions.add_parser("evaluate", help="Calculate COCO AP, recall and detection errors")
    path_argument(evaluate, "manifest")
    path_argument(evaluate, "checkpoint")
    path_argument(evaluate, "output")
    evaluate.add_argument("--split", choices=("val", "test"), required=True)
    evaluate.add_argument("--threshold", type=float, default=0.25)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")

    export = actions.add_parser("export", help="Export the checkpoint to ONNX")
    path_argument(export, "checkpoint")
    path_argument(export, "output")

    compare = actions.add_parser("compare", help="Compare PyTorch and ONNX predictions and CPU latency")
    path_argument(compare, "manifest")
    path_argument(compare, "checkpoint")
    path_argument(compare, "onnx")
    path_argument(compare, "output")
    compare.add_argument("--samples", type=int, default=60)
    compare.add_argument("--threads", type=int, default=1)
    compare.add_argument("--threshold", type=float, default=0.25)

    args = vars(parser.parse_args())
    action = args.pop("action")
    if action == "prepare":
        prepared = prepare(**args)
        counts = {split: sum(row["split"] == split for row in prepared["images"]) for split in ("train", "val", "test")}
        print(f"Wrote {args['manifest']} ({counts})")
    elif action == "train":
        from .model import train
        train(**args)
    elif action == "export":
        from .model import export_model
        export_model(**args)
    elif action == "evaluate":
        from .evaluation import evaluate
        evaluate(**args)
    elif action == "compare":
        from .evaluation import compare
        if not compare(**args)["passed"]:
            parser.exit(1, "Runtime parity failed; see the JSON report.\n")


if __name__ == "__main__":
    main()
