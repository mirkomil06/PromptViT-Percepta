import torch
import torch.nn as nn
import timm


class ViTVPTDeep(nn.Module):
    """
    Vision Transformer + deep Visual Prompt Tuning (VPT-Deep).

    - Backbone: vit_base_patch16_224 from timm
    - Backbone is FROZEN (no gradient)
    - Learnable prompt tokens per Transformer layer
    - At EACH layer i:
        input = [CLS, P_i1..P_iP, patches]
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

        embed_dim = self.backbone.embed_dim      # 768 for ViT-B/16
        self.num_prompts = num_prompts
        self.num_layers = len(self.backbone.blocks)  # usually 12

        # 2. Remove original head, we'll use our own classifier
        self.backbone.head = nn.Identity()

        # 3. Freeze all backbone parameters (VPT idea)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 4. Learnable deep prompts:
        # shape: (num_layers, num_prompts, embed_dim)
        self.deep_prompt_embeddings = nn.Parameter(
            torch.zeros(self.num_layers, num_prompts, embed_dim)
        )
        nn.init.normal_(self.deep_prompt_embeddings, std=0.02)

        # 5. New classification head (trainable)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Forward pass with deep prompts:

        Initial tokens:
            x = patch_embed(image)                 -> (B, N, D)
            cls = cls_token                        -> (B, 1, D)
            tokens = [CLS, patch_1..patch_N]       -> (B, 1+N, D)
            add pos_embed (only for CLS + patches)
            pos_drop

        For each layer i:
            - drop old prompts (if i > 0)
            - insert layer-specific prompts P_i after CLS
            - run transformer block i
        """
        B = x.shape[0]

        # --- Patch embedding ---
        x = self.backbone.patch_embed(x)  # (B, N, D)

        # CLS token
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)  # (B, 1, D)

        # concat [CLS] + patches → (B, 1+N, D)
        x = torch.cat((cls_tokens, x), dim=1)

        # add position embeddings (only for CLS + patches)
        # pos_embed shape: (1, 1+N, D)
        x = x + self.backbone.pos_embed

        # apply dropout
        x = self.backbone.pos_drop(x)

        # --- Transformer blocks with deep prompts ---
        for layer_idx, blk in enumerate(self.backbone.blocks):
            # x currently has:
            #   - for layer 0: [CLS, patch_1..patch_N]         (1+N tokens)
            #   - for layer >0: [CLS, P_(i-1)1..P_(i-1)P, patch_1..patch_N]
            #                   (1 + num_prompts + N tokens)

            cls_tok = x[:, :1, :]  # (B, 1, D)

            if layer_idx == 0:
                # no prompts yet, all after CLS are patches
                patch_tok = x[:, 1:, :]  # (B, N, D)
            else:
                # drop previous layer's prompts (positions 1..P),
                # keep only patches after them
                patch_tok = x[:, 1 + self.num_prompts:, :]  # (B, N, D)

            # get prompts for THIS layer: (num_prompts, D)
            P_i = self.deep_prompt_embeddings[layer_idx]  # (P, D)
            # expand for batch: (B, P, D)
            P_i = P_i.unsqueeze(0).expand(B, -1, -1)

            # new sequence: [CLS, P_i1..P_iP, patch_1..patch_N]
            x = torch.cat((cls_tok, P_i, patch_tok), dim=1)  # (B, 1+P+N, D)

            # standard ViT block
            x = blk(x)

        # final layer norm
        x = self.backbone.norm(x)

        # take CLS output (still at index 0)
        cls_out = x[:, 0]

        # classification head
        logits = self.head(cls_out)
        return logits
