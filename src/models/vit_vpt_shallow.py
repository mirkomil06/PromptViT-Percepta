import torch
import torch.nn as nn
import timm


class ViTVPTShallow(nn.Module):
    """
    Vision Transformer + shallow Visual Prompt Tuning (VPT).

    - Backbone: vit_base_patch16_224 from timm
    - Backbone заморожен (no grad)
    - Learnable prompt tokens вставляются ПОСЛЕ [CLS], ПЕРЕД патчами
    - Обучаются только промпты + классификационный head
    """
    def __init__(
        self,
        num_classes: int = 200,
        num_prompts: int = 10,
        pretrained: bool = True,
    ):
        super().__init__()

        # 1. Загружаем ViT-B/16 backbone
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            drop_rate=0.1,        # такой же dropout, как у baseline
            drop_path_rate=0.1,   # такой же stochastic depth
        )

        embed_dim = self.backbone.embed_dim  # 768 для ViT-B/16

        # 2. Убираем исходный head, используем свой классификатор
        self.backbone.head = nn.Identity()

        # 3. Замораживаем все параметры backbone (VPT идея)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 4. Обучаемые prompt-токены (одни для всех картинок, в forward расширяем по batch)
        self.num_prompts = num_prompts
        self.prompt_embeddings = nn.Parameter(
            torch.zeros(1, num_prompts, embed_dim)
        )
        nn.init.normal_(self.prompt_embeddings, std=0.02)

        # 5. Новый классификационный head (trainable)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward с shallow VPT:

        [CLS] + patches  -> add pos_embed
        -> ВСТАВЛЯЕМ P prompt-токенов ПОСЛЕ CLS
        -> прогоняем через блоки ViT
        -> берём CLS и подаём в head
        """
        B = x.shape[0]

        # Patch embedding
        x = self.backbone.patch_embed(x)  # (B, N, D)

        # CLS token
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)  # (B, 1, D)

        # concat [CLS] + patches → (B, 1+N, D)
        x = torch.cat((cls_tokens, x), dim=1)

        # добавляем position embeddings (только для CLS + patches)
        # pos_embed shape: (1, 1+N, D)
        x = x + self.backbone.pos_embed

        # создаём prompt-токены для этого batch’а
        prompt_tokens = self.prompt_embeddings.expand(B, -1, -1)  # (B, P, D)

        # хотим: [CLS, P1..Pp, patch_1..patch_N]
        x = torch.cat(
            (x[:, :1, :], prompt_tokens, x[:, 1:, :]),
            dim=1
        )  # (B, 1+P+N, D)

        # стандартный ViT forward через блоки
        x = self.backbone.pos_drop(x)
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)

        # берём CLS (всё ещё индекс 0)
        cls_out = x[:, 0]

        # классификационный head
        logits = self.head(cls_out)
        return logits
