import torch.nn as nn
import timm


class ViTBaseline(nn.Module):
    """
    Базовая модель ViT-B/16 для CUB-200.
    Используем timm.create_model и меняем head под 200 классов.
    """
    def __init__(self, num_classes: int = 200, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained
        )
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
