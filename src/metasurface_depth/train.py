"""Single-GPU mixed-dataset training with explicit configuration and safe checkpoints."""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import EncodedDataset, assert_disjoint
from .evaluate import evaluate_model
from .io import new_directory, sha256, write_json
from .metrics import training_loss
from .model import MetasurfaceDepth, load_model

DEFAULTS = dict(
    model="small",
    steps=80000,
    warmup_steps=1000,
    batch_size=8,
    crop_size=518,
    seed=0,
    lr_backbone=3e-6,
    lr_head=1e-5,
    weight_decay=0.01,
    decay_steps=10000,
    decay_gamma=0.8,
    validate_every=1000,
    save_every=2000,
    log_every=10,
    augmentation_start=2000,
    augmentation_probability=0.4,
    selection_metric="original",
    selection_dataset=0,
)


def optimizer_for(model, cfg, warmup):
    backbone, head = [], []
    for name, param in model.named_parameters():
        if name.startswith("pretrained."):
            backbone.append(param)
        else:
            param.requires_grad_(not warmup)
            head.append(param)
    opt = torch.optim.AdamW(
        [
            dict(params=backbone, lr=cfg["lr_backbone"]),
            dict(params=head, lr=0.0 if warmup else cfg["lr_head"]),
        ],
        weight_decay=cfg["weight_decay"],
    )
    return opt, torch.optim.lr_scheduler.StepLR(
        opt, step_size=cfg["decay_steps"], gamma=cfg["decay_gamma"]
    )


def save_checkpoint(path, model, config, step, optimizer, scheduler, best):
    # Immediate serialization avoids retaining live best-weight tensor references.
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    payload = dict(
        format_version=1,
        state_dict=state,
        config=dict(variant=config["model"], depth_range_m=[0.2, 1.2]),
        training=dict(
            step=step,
            settings=config,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            best=best,
        ),
    )
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path, help="New run directory")
    init = p.add_mutually_exclusive_group(required=True)
    init.add_argument("--weights", type=Path, help="Fine-tune a project export")
    init.add_argument(
        "--pretrained", type=Path, help="Initialize from DAV2 metric-Hypersim state dict"
    )
    init.add_argument(
        "--resume", type=Path, help="Resume model/optimizer/step into a NEW run directory"
    )
    init.add_argument("--random-init", action="store_true", help="Smoke tests only")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--stop-after",
        type=int,
        help="Stop this invocation at an absolute step; useful for resume checks",
    )
    args = p.parse_args()
    raw = json.loads(args.config.read_text())
    unknown = set(raw) - set(DEFAULTS) - {"train", "validation"}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    cfg = {**DEFAULTS, **raw}
    for key in (
        "steps",
        "batch_size",
        "crop_size",
        "decay_steps",
        "save_every",
        "validate_every",
        "log_every",
    ):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["warmup_steps"] < 0 or cfg["crop_size"] % 14:
        raise ValueError("Invalid warmup_steps or crop_size")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    config_dir = args.config.resolve().parent
    train_specs, val_specs = cfg.get("train", []), cfg.get("validation", [])
    if not train_specs:
        raise ValueError("At least one training manifest is required")
    datasets = [
        EncodedDataset(config_dir / item["manifest"], cfg["crop_size"], True)
        for item in train_specs
    ]
    validation = [EncodedDataset(config_dir / item["manifest"]) for item in val_specs]
    if cfg["selection_metric"] not in ("original", "mae"):
        raise ValueError("selection_metric must be original or mae")
    if validation and not 0 <= cfg["selection_dataset"] < len(validation):
        raise ValueError("selection_dataset is outside the validation list")
    assert_disjoint(datasets, validation)
    if any(len(d) < cfg["batch_size"] for d in datasets):
        raise ValueError("Every training dataset must contain at least one full batch")
    probs = np.array([item.get("weight", 1.0) for item in train_specs], dtype=np.float32)
    if not np.isfinite(probs).all() or (probs <= 0).any():
        raise ValueError("Dataset sampling weights must be finite and positive")
    probs /= probs.sum()
    loaders = [
        DataLoader(d, batch_size=cfg["batch_size"], shuffle=True, num_workers=0, drop_last=True)
        for d in datasets
    ]
    model = MetasurfaceDepth(cfg["model"])
    initialization = dict(kind="random-smoke-only")
    resumed = None
    if args.weights or args.resume:
        path = args.weights or args.resume
        model, _ = load_model(path, "cpu", cfg["model"])
        initialization = dict(kind="resume" if args.resume else "project", sha256=sha256(path))
        if args.resume:
            resumed = torch.load(path, map_location="cpu", weights_only=True)["training"]
            if resumed["settings"] != cfg:
                raise ValueError("Resume requires the same training config")
    elif args.pretrained:
        state = torch.load(args.pretrained, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(k.startswith("pretrained.") for k in missing):
            raise ValueError("DAV2 checkpoint does not match the selected backbone")
        initialization = dict(
            kind="DAV2", sha256=sha256(args.pretrained), missing_new_parameters=missing
        )
    model.to(args.device)
    start = resumed["step"] if resumed else 0
    warmup = cfg["warmup_steps"] > 0 and start < cfg["warmup_steps"]
    optimizer, scheduler = optimizer_for(model, cfg, warmup)
    best = resumed["best"] if resumed else None
    if resumed:
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
    output = new_directory(args.output)
    write_json(output / "config.json", cfg)
    write_json(output / "initialization.json", initialization)
    write_json(
        output / "manifests.json",
        {
            item["manifest"]: sha256(config_dir / item["manifest"])
            for item in train_specs + val_specs
        },
    )
    iterators = [iter(loader) for loader in loaders]
    from .augmentation import Augmentation

    augmentation = Augmentation(cfg["augmentation_probability"], 1.2, 0.06, 2.0, 0.2, 0.4, 25)
    total = cfg["steps"] + cfg["warmup_steps"]
    if start >= total:
        raise ValueError("Checkpoint already reached the configured total steps")
    end = min(total, args.stop_after) if args.stop_after else total
    if end <= start:
        raise ValueError("stop-after must be greater than the resumed step")
    with (output / "metrics.jsonl").open("x") as log:
        for step in range(start + 1, end + 1):
            if warmup and step == cfg["warmup_steps"]:
                optimizer, scheduler = optimizer_for(model, cfg, False)
                warmup = False
            for spec, dataset in zip(train_specs, datasets):
                dataset.augmentation = (
                    augmentation
                    if (step >= cfg["augmentation_start"] and spec.get("augment", True))
                    else None
                )
            index = int(np.random.choice(len(datasets), p=probs))
            try:
                image, depth = next(iterators[index])
            except StopIteration:
                iterators[index] = iter(loaders[index])
                image, depth = next(iterators[index])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(image.to(args.device))
            loss = training_loss(prediction, depth.to(args.device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step}")
            loss.backward()
            optimizer.step()
            scheduler.step()
            row = dict(step=step, loss=float(loss.detach()), dataset=index)
            if validation and (step % cfg["validate_every"] == 0 or step == end):
                reports = [evaluate_model(model, data, args.device) for data in validation]
                selected = reports[cfg["selection_dataset"]]["summary"]["full"]
                values = {key: value["mean"] for key, value in selected.items()}
                score = (
                    values["mae"]
                    if cfg["selection_metric"] == "mae"
                    else (
                        math.sqrt(values["mae"] * values["abs_rel"] * values["rmse"])
                        * (1 - values["delta_0_5"])
                    )
                )
                row["validation_mae"] = values["mae"]
                row["selection_score"] = score
                if best is None or score < best:
                    best = score
                    save_checkpoint(
                        output / "best.pth", model, cfg, step, optimizer, scheduler, best
                    )
            if step % cfg["save_every"] == 0 or step == end:
                save_checkpoint(output / "last.pth", model, cfg, step, optimizer, scheduler, best)
            if step % cfg["log_every"] == 0 or step == end:
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(row, flush=True)


if __name__ == "__main__":
    main()
