import os
import yaml
import random
import argparse
import platform

import torch
from torch.utils.data import DataLoader

from src.datasets.cub200 import CUB200Dataset
from src.models.vit_vpt_shallow import ViTVPTShallow
from src.training.trainer_baseline import Trainer


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ViT-B/16 with shallow VPT on CUB-200"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., src/configs/cub_vpt_shallow.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("experiment_name", "cub_vpt_shallow")
    output_dir = cfg.get("output_dir", os.path.join("outputs", exp_name))

    dataset_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    # 2. Seed
    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    # 3. Device (from config if given, else auto)
    cfg_device = train_cfg.get("device", None)
    if cfg_device is not None:
        device = torch.device(cfg_device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Main] VPT: Using device: {device}")

    # 4. Dataset & DataLoaders
    root = dataset_cfg.get("root", "data/cub200/CUB_200_2011")
    image_size = dataset_cfg.get("image_size", 224)

    train_dataset = CUB200Dataset(
        root=root,
        train=True,
        image_size=image_size,
    )
    val_dataset = CUB200Dataset(
        root=root,
        train=False,
        image_size=image_size,
    )

    batch_size = train_cfg.get("batch_size", 16)
    num_workers = train_cfg.get("num_workers", 0)

    # Windows safety: no DataLoader multiprocessing (avoids shm.dll issues)
    if platform.system() == "Windows":
        if num_workers != 0:
            print(
                f"[Main] VPT: Overriding num_workers={num_workers} -> 0 on Windows "
                "to avoid CUDA DLL spawn errors."
            )
        num_workers = 0

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 5. Model (VPT shallow)
    num_classes = model_cfg.get("num_classes", 200)
    num_prompts = model_cfg.get("num_prompts", 10)
    pretrained = model_cfg.get("pretrained", True)

    model = ViTVPTShallow(
        num_classes=num_classes,
        num_prompts=num_prompts,
        pretrained=pretrained,
    )

    # 6. Trainer + TensorBoard
    lr = train_cfg.get("lr", 3e-4)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    num_epochs = train_cfg.get("num_epochs", 30)
    label_smoothing = train_cfg.get("label_smoothing", 0.1)

    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        output_dir=output_dir,
        log_dir=log_dir,
        label_smoothing=label_smoothing,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
