import os
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        output_dir: str = "outputs/cub_baseline",
        log_dir: Optional[str] = None,
        label_smoothing: float = 0.1,
    ):
        """
        Basic training loop wrapper for classification.

        Args:
            model: nn.Module to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            device: torch.device ("cuda" or "cpu").
            lr: learning rate for AdamW.
            weight_decay: weight decay for AdamW.
            num_epochs: number of training epochs.
            output_dir: directory for checkpoints and logs.
            log_dir: directory for TensorBoard logs (defaults to output_dir/logs).
            label_smoothing: epsilon for label smoothing in CrossEntropyLoss.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs

        # Loss with optional label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.best_val_acc = 0.0

        # TensorBoard writer
        if log_dir is None:
            log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(self.train_loader, desc=f"Epoch {epoch} [train]")

        for images, labels in loop:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix({
                "loss": loss.item(),
                "acc": f"{100.0 * correct / total:.2f}%",
            })

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        return {"loss": epoch_loss, "acc": epoch_acc}

    @torch.no_grad()
    def evaluate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(self.val_loader, desc=f"Epoch {epoch} [val]")

        for images, labels in loop:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix({
                "loss": loss.item(),
                "acc": f"{100.0 * correct / total:.2f}%",
            })

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        return {"loss": epoch_loss, "acc": epoch_acc}

    def fit(self):
        for epoch in range(1, self.num_epochs + 1):
            train_stats = self.train_one_epoch(epoch)
            val_stats = self.evaluate(epoch)

            # Log to TensorBoard
            self.writer.add_scalar("Loss/train", train_stats["loss"], epoch)
            self.writer.add_scalar("Loss/val", val_stats["loss"], epoch)
            self.writer.add_scalar("Accuracy/train", train_stats["acc"], epoch)
            self.writer.add_scalar("Accuracy/val", val_stats["acc"], epoch)

            print(
                f"[Epoch {epoch}] "
                f"train_loss={train_stats['loss']:.4f}, "
                f"train_acc={train_stats['acc']:.2f} | "
                f"val_loss={val_stats['loss']:.4f}, "
                f"val_acc={val_stats['acc']:.2f}"
            )

            # save best checkpoint based on val_acc
            if val_stats["acc"] > self.best_val_acc:
                self.best_val_acc = val_stats["acc"]
                ckpt_path = os.path.join(self.output_dir, "best_model.pth")
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"[Trainer] Saved new best model to {ckpt_path}")

        # Close TensorBoard writer
        self.writer.close()
        print("[Trainer] Training finished.")