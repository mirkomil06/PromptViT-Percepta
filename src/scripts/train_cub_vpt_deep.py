import os
import yaml
import random
import argparse

import torch
from torch.utils.data import DataLoader

from src.datasets.cub200 import CUB200Dataset
from src.models.vit_vpt_deep import ViTVPTDeep
from src.training.trainer_baseline import Trainer


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/cub_vpt_deep.yaml",
        help="Path to YAML config file"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # === LOAD CONFIG ===
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    experiment_name = cfg.get("experiment_name", "cub_vpt_deep")
    output_dir = cfg.get("output_dir", f"outputs/{experiment_name}")

    dataset_cfg = cfg["dataset"]
    train_cfg = cfg["training"]
    model_cfg = cfg.get("model", {})

    set_seed(train_cfg.get("seed", 42))

    # === DEVICE ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] VPT-Deep: Using device: {device}")

    # === DATASETS ===
    root = dataset_cfg["root"]
    image_size = dataset_cfg.get("image_size", 224)

    train_dataset = CUB200Dataset(root=root, train=True, image_size=image_size)
    val_dataset = CUB200Dataset(root=root, train=False, image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=True
    )

    # === MODEL (VPT-Deep) ===
    num_classes = model_cfg.get("num_classes", 200)
    num_prompts = model_cfg.get("num_prompts", 10)
    pretrained = model_cfg.get("pretrained", True)

    model = ViTVPTDeep(
        num_classes=num_classes,
        num_prompts=num_prompts,
        pretrained=pretrained,
    )

    # === TRAINER ===
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])
    num_epochs = int(train_cfg["num_epochs"])

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        output_dir=output_dir
    )

    trainer.fit()


if __name__ == "__main__":
    main()
