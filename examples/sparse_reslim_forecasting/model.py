"""Minimal Sparse-Reslim model for deterministic weather forecasting.

This example intentionally depends only on PyTorch.  It demonstrates the
dense-sparse-dense token route used by Sparse-Reslim without pulling the full
ECCV research code into ORBIT-2.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class SparseReslim(nn.Module):
    """A compact Sparse-Reslim forecasting backbone.

    The model embeds every weather variable independently, aggregates variables
    at each spatial location, and uses all tokens in the early and late
    Transformer blocks.  Only ``keep_ratio`` of the tokens enter the middle
    blocks.  Their residual updates are scattered back onto the dense grid, so
    skipped tokens follow an identity path.

    Inputs may be ``[batch, time, variable, height, width]`` or
    ``[batch, variable, height, width]``.  Forecasts have shape
    ``[batch, output_variable, height, width]``.
    """

    def __init__(
        self,
        variables: Sequence[str],
        output_variables: Sequence[str],
        img_size: tuple[int, int],
        *,
        history: int = 1,
        patch_size: int = 2,
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        keep_ratio: float = 0.25,
        num_dense_early: int = 1,
        num_sparse_middle: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        if num_dense_early < 0 or num_sparse_middle < 0:
            raise ValueError("block counts must be non-negative")
        if num_dense_early + num_sparse_middle > depth:
            raise ValueError("dense early + sparse middle blocks exceed depth")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")

        height, width = img_size
        if height % patch_size or width % patch_size:
            raise ValueError("img_size must be divisible by patch_size")

        self.variables = tuple(variables)
        self.output_variables = tuple(output_variables)
        unknown_outputs = set(self.output_variables) - set(self.variables)
        if unknown_outputs:
            raise ValueError(
                "output variables must also be inputs for the residual path: "
                f"{sorted(unknown_outputs)}"
            )

        self.variable_to_index = {
            variable: index for index, variable in enumerate(self.variables)
        }
        self.history = history
        self.img_size = (height, width)
        self.patch_size = patch_size
        self.keep_ratio = keep_ratio
        self.num_dense_early = num_dense_early
        self.num_sparse_middle = num_sparse_middle
        self.num_dense_late = depth - num_dense_early - num_sparse_middle

        self.patch_embeds = nn.ModuleDict(
            {
                variable: nn.Conv2d(
                    history,
                    embed_dim,
                    kernel_size=patch_size,
                    stride=patch_size,
                )
                for variable in self.variables
            }
        )
        grid_height = height // patch_size
        grid_width = width // patch_size
        self.grid_size = (grid_height, grid_width)
        self.num_patches = grid_height * grid_width

        self.variable_embed = nn.Parameter(
            torch.zeros(1, 1, len(self.variables), embed_dim)
        )
        self.variable_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.variable_aggregate = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.position_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        hidden_dim = int(embed_dim * mlp_ratio)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        output_channels = len(self.output_variables)
        self.decoder = nn.Linear(
            embed_dim, output_channels * patch_size * patch_size
        )
        residual_width = max(16, embed_dim // 4)
        self.residual_path = nn.Sequential(
            nn.Conv2d(output_channels, residual_width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(residual_width, output_channels, kernel_size=3, padding=1),
        )

        self.last_sparse_token_count = self.num_patches
        nn.init.trunc_normal_(self.variable_embed, std=0.02)
        nn.init.trunc_normal_(self.variable_query, std=0.02)
        nn.init.trunc_normal_(self.position_embed, std=0.02)

    def _embed(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 4:
            inputs = inputs.unsqueeze(1)
        if inputs.ndim != 5:
            raise ValueError(
                "inputs must have shape [B,T,V,H,W] or [B,V,H,W]"
            )

        batch, history, variables, height, width = inputs.shape
        if history != self.history:
            raise ValueError(f"expected {self.history} history steps, got {history}")
        if variables != len(self.variables):
            raise ValueError(
                f"expected {len(self.variables)} variables, got {variables}"
            )
        if (height, width) != self.img_size:
            raise ValueError(f"expected spatial size {self.img_size}, got {(height, width)}")

        # Each variable is embedded from all history channels independently.
        embedded = []
        for variable_index, variable in enumerate(self.variables):
            patches = self.patch_embeds[variable](inputs[:, :, variable_index])
            patches = patches.flatten(2).transpose(1, 2)
            embedded.append(patches)

        # [B, L, V, D], followed by attention over variables at every location.
        tokens = torch.stack(embedded, dim=2) + self.variable_embed
        tokens = tokens.flatten(0, 1)
        query = self.variable_query.expand(tokens.shape[0], -1, -1)
        tokens, _ = self.variable_aggregate(
            query, tokens, tokens, need_weights=False
        )
        tokens = tokens.squeeze(1).unflatten(0, (batch, self.num_patches))
        return tokens + self.position_embed

    def _route(self, tokens: torch.Tensor) -> torch.Tensor:
        block_index = 0
        for _ in range(self.num_dense_early):
            tokens = self.blocks[block_index](tokens)
            block_index += 1

        if self.num_sparse_middle:
            batch, sequence_length, embed_dim = tokens.shape
            num_keep = max(1, int(sequence_length * self.keep_ratio))
            self.last_sparse_token_count = num_keep

            # Parameter-free routing: independently sample a token subset per item.
            indices = torch.rand(
                batch, sequence_length, device=tokens.device
            ).topk(num_keep, dim=1, sorted=False).indices
            indices = indices.sort(dim=1).values
            expanded_indices = indices.unsqueeze(-1).expand(-1, -1, embed_dim)
            sparse_tokens = tokens.gather(1, expanded_indices)
            sparse_input = sparse_tokens

            for _ in range(self.num_sparse_middle):
                sparse_tokens = self.blocks[block_index](sparse_tokens)
                block_index += 1

            # Restore the dense grid with a residual update.  Unselected tokens
            # receive a zero delta and therefore keep their original state.
            delta = torch.zeros_like(tokens)
            delta.scatter_(1, expanded_indices, sparse_tokens - sparse_input)
            tokens = tokens + delta
        else:
            self.last_sparse_token_count = tokens.shape[1]

        for _ in range(self.num_dense_late):
            tokens = self.blocks[block_index](tokens)
            block_index += 1
        return self.norm(tokens)

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch = patches.shape[0]
        grid_height, grid_width = self.grid_size
        patch = self.patch_size
        channels = len(self.output_variables)
        patches = patches.reshape(
            batch, grid_height, grid_width, channels, patch, patch
        )
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        return patches.reshape(
            batch, channels, grid_height * patch, grid_width * patch
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 4:
            inputs = inputs.unsqueeze(1)
        output_indices = [
            self.variable_to_index[variable] for variable in self.output_variables
        ]
        residual = self.residual_path(inputs[:, -1, output_indices])
        tokens = self._route(self._embed(inputs))
        forecast = self._unpatchify(self.decoder(tokens))
        return forecast + residual
