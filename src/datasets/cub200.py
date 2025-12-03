import os
from typing import Tuple, List

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CUB200Dataset(Dataset):
    """
    PyTorch Dataset для CUB_200_2011.
    Использует официальные файлы:
      - images.txt
      - image_class_labels.txt
      - train_test_split.txt
    """
    def __init__(self, root: str, train: bool = True, image_size: int = 224):
        """
        :param root: путь до папки CUB_200_2011 (где лежат images.txt и т.д.)
        :param train: True -> train split, False -> test/val split
        :param image_size: базовый размер для ресайза/кропа
        """
        self.root = root
        self.train = train
        self.image_size = image_size

        self.images_dir = os.path.join(root, "images")
        images_txt = os.path.join(root, "images.txt")
        labels_txt = os.path.join(root, "image_class_labels.txt")
        split_txt = os.path.join(root, "train_test_split.txt")

        # id → path / label / split
        self.id_to_path = {}
        with open(images_txt, "r") as f:
            for line in f:
                image_id_str, rel_path = line.strip().split()
                self.id_to_path[int(image_id_str)] = rel_path

        self.id_to_label = {}
        with open(labels_txt, "r") as f:
            for line in f:
                image_id_str, label_str = line.strip().split()
                # В CUB классы начинаются с 1, переводим в 0..199
                self.id_to_label[int(image_id_str)] = int(label_str) - 1

        self.id_to_is_train = {}
        with open(split_txt, "r") as f:
            for line in f:
                image_id_str, is_train_str = line.strip().split()
                self.id_to_is_train[int(image_id_str)] = (int(is_train_str) == 1)

        # Собираем список (полный путь, label) только для нужного split
        self.samples: List[Tuple[str, int]] = []
        for image_id, rel_path in self.id_to_path.items():
            if self.id_to_is_train[image_id] == self.train:
                label = self.id_to_label[image_id]
                full_path = os.path.join(self.images_dir, rel_path)
                self.samples.append((full_path, label))

        # Трансформации: разные для train и val
        if self.train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],   # стандартные ImageNet-статистики
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            # Для валидации/теста — стабильный Resize + CenterCrop
            self.transform = transforms.Compose([
                transforms.Resize(int(self.image_size * 1.1)),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        print(
            f"[CUB200Dataset] root={root}, train={train}, "
            f"samples={len(self.samples)}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
