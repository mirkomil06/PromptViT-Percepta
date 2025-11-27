import torch
import torch.nn as nn
import timm


class ViTVPTShallow(nn.Module):
    """
    Vision Transformer + shallow Visual Prompt Tuning.

    - Backbone: vit_base_patch16_224 from timm
    - Backbone is FROZEN (no gradient)
    - Learnable prompt tokens are prepended after [CLS]
    - Only prompts + classification head are trained
    """
    def __init__(
        self,
        num_classes: int = 200,
        num_prompts: int = 10,
        pretrained: bool = True,
    ):
        super().__init__()

        # 1. Load pretrained ViT backbone
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained
        )

        embed_dim = self.backbone.embed_dim  # 768 for ViT-B/16

        # 2. Remove original head, we'll use our own classifier
        self.backbone.head = nn.Identity()

        # 3. Freeze all backbone parameters (VPT idea)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 4. Learnable prompt tokens (same for all images, expanded in forward)
        self.num_prompts = num_prompts
        self.prompt_embeddings = nn.Parameter(
            torch.zeros(1, num_prompts, embed_dim)
        )
        nn.init.normal_(self.prompt_embeddings, std=0.02)

        # 5. New classification head (trainable)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Forward pass with shallow prompts:

        [CLS] + patches  -> add pos_embed
        insert prompts after [CLS]:

        [CLS] , P1, P2, ..., Pp, patch_1, ..., patch_N
        """
        B = x.shape[0]

        # Patch embedding
        x = self.backbone.patch_embed(x)  # (B, N, D)

        # CLS token
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)  # (B, 1, D)

        # concat [CLS] + patches → (B, 1+N, D)
        x = torch.cat((cls_tokens, x), dim=1)

        # add position embeddings (only for CLS + patches)
        # pos_embed shape: (1, 1+N, D)
        x = x + self.backbone.pos_embed

        # create prompts for this batch
        prompt_tokens = self.prompt_embeddings.expand(B, -1, -1)  # (B, P, D)

        # insert prompts AFTER CLS, BEFORE patches
        # x: [CLS, patch_1, patch_2, ...]
        # we want: [CLS, P..., patch_1, patch_2, ...]
        x = torch.cat(
            (x[:, :1, :], prompt_tokens, x[:, 1:, :]),
            dim=1
        )  # (B, 1+P+N, D)

        # standard ViT forward over transformer blocks
        x = self.backbone.pos_drop(x)
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)

        # take CLS output (still at index 0)
        cls_out = x[:, 0]

        # classification head
        logits = self.head(cls_out)
        return logits
