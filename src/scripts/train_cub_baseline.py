import os
import yaml
import random
import argparse

import torch
from torch.utils.data import DataLoader

from src.datasets.cub200 import CUB200Dataset
from src.models.vit_baseline import ViTBaseline
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
        default="src/configs/cub_baseline.yaml",
        help="Path to YAML config file"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)


    experiment_name = cfg.get("experiment_name", "cub_vit_baseline")
    output_dir = cfg.get("output_dir", f"outputs/{experiment_name}")

    dataset_cfg = cfg["dataset"]
    train_cfg = cfg["training"]

    set_seed(train_cfg.get("seed", 42))

    device = torch.device("cpu")
    print("[Main] Running on CPU only.")

    # ДАТАСЕТЫ
    root = dataset_cfg["root"]
    image_size = dataset_cfg.get("image_size", 224)

    train_dataset = CUB200Dataset(root=root, train=True, image_size=image_size)
    val_dataset = CUB200Dataset(root=root, train=False, image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True
    )

    # МОДЕЛЬ
    model = ViTBaseline(num_classes=200, pretrained=True)

    # ТРЕНЕР
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        num_epochs=train_cfg["num_epochs"],
        output_dir=output_dir
    )

    trainer.fit()


if __name__ == "__main__":
    main()
