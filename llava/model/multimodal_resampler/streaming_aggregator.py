"""
StreamingStateAggregator — LLaVA-NeXT integration adapter.

Adapted from rep_sim/streaming_aggregator_draft.py (webdataset branch).
Key changes vs. the original:
  - _flatten_patch_dim inlined (no dep on aggregators module)
  - `return_state_tokens` flag added to forward()
  - `hidden_size` and `config` properties for LLaVA builder / trainer
  - `num_layers` stored as instance attribute
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers (inlined from rep_sim/aggregators.py)
# ---------------------------------------------------------------------------

def _flatten_patch_dim(
    frame_embeddings: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
):
    """
    If frame_embeddings has shape (B, T, N, D) or (T, N, D), flatten the
    T×N token dimension into T*N so the rest of the module sees a flat
    sequence.  The mask (B, T) is expanded to (B, T*N) accordingly.

    Shapes that are already 2D or 3D (no patch dim) pass through unchanged.
    """
    if frame_embeddings.ndim == 4:
        B, T, N, D = frame_embeddings.shape
        frame_embeddings = frame_embeddings.reshape(B, T * N, D)
        if mask is not None:
            mask = mask.unsqueeze(-1).expand(B, T, N).reshape(B, T * N)
    elif frame_embeddings.ndim == 3:
        # Could be (B, T, D) — no patch dim — or unbatched (T, N, D)
        # Disambiguate: if called after unsqueeze(0) for unbatched input we
        # already have (1, T, D); just pass through.
        pass
    return frame_embeddings, mask


# ---------------------------------------------------------------------------
# Positional encodings
# ---------------------------------------------------------------------------

class RoPE1D(nn.Module):
    def __init__(self, freq: float = 10000.0):
        super().__init__()
        self.freq = freq
        self._cache: dict = {}

    def _get_cos_sin(self, head_dim, max_pos, dtype, device):
        key = (head_dim, max_pos, dtype, device)
        if key not in self._cache:
            inv_freq = 1.0 / (
                self.freq ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
            )
            t = torch.arange(max_pos + 1, device=device, dtype=torch.float32)
            freqs = torch.cat([torch.outer(t, inv_freq)] * 2, dim=-1)
            self._cache[key] = (freqs.cos().to(dtype), freqs.sin().to(dtype))
        return self._cache[key]

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, tokens, positions):
        B, H, N, head_dim = tokens.shape
        pos = positions.long()
        if pos.ndim == 1:
            pos = pos.unsqueeze(0).expand(B, -1)
        cos, sin = self._get_cos_sin(head_dim, int(pos.amax().item()), tokens.dtype, tokens.device)
        cos_tok = cos[pos][:, None, :, :]
        sin_tok = sin[pos][:, None, :, :]
        return tokens * cos_tok + self._rotate_half(tokens) * sin_tok


class RoPE2D(nn.Module):
    def __init__(self, freq: float = 100.0):
        super().__init__()
        self.freq = freq
        self._cache: dict = {}

    def _get_cos_sin(self, half_dim, positions_1d, out_dtype, device):
        min_pos = int(positions_1d.amin().item())
        max_pos = int(positions_1d.amax().item())
        key = (half_dim, min_pos, max_pos, device, out_dtype)
        if key not in self._cache:
            inv_freq = 1.0 / (
                self.freq ** (torch.arange(0, half_dim, 2, device=device, dtype=torch.float32) / half_dim)
            )
            t = torch.arange(min_pos, max_pos + 1, device=device, dtype=torch.float32)
            freqs = torch.cat([torch.outer(t, inv_freq)] * 2, dim=-1)
            self._cache[key] = (freqs.cos().to(out_dtype), freqs.sin().to(out_dtype))
        return self._cache[key]

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, tokens, positions):
        half_dim = tokens.shape[-1] // 2
        B = tokens.shape[0]
        rope_dtype = tokens.dtype
        device = tokens.device

        row_cos, row_sin = self._get_cos_sin(half_dim, positions[..., 0], rope_dtype, device)
        col_cos, col_sin = self._get_cos_sin(half_dim, positions[..., 1], rope_dtype, device)

        pos_row = (positions[..., 0] - positions[..., 0].amin()).long()
        pos_col = (positions[..., 1] - positions[..., 1].amin()).long()

        row_cos = row_cos[pos_row][:, None, :, :]
        row_sin = row_sin[pos_row][:, None, :, :]
        col_cos = col_cos[pos_col][:, None, :, :]
        col_sin = col_sin[pos_col][:, None, :, :]

        y, x = tokens.chunk(2, dim=-1)
        y = y * row_cos + self._rotate_half(y) * row_sin
        x = x * col_cos + self._rotate_half(x) * col_sin
        return torch.cat([y, x], dim=-1)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, rope=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.rope = rope

    def forward(self, x, pos=None, key_padding_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros((B, 1, N, N), device=x.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~key_padding_mask.bool()[:, None, None, :], torch.finfo(q.dtype).min)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, rope_k=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_out = nn.Linear(dim, dim)
        self.rope_k = rope_k

    def forward(self, x, context, x_pos=None, ctx_pos=None, ctx_key_padding_mask=None):
        B, N, C = x.shape
        M = context.shape[1]
        q = self.proj_q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.proj_k(context).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.proj_v(context).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        if self.rope_k is not None and ctx_pos is not None:
            k = self.rope_k(k, ctx_pos)
        attn_mask = None
        if ctx_key_padding_mask is not None:
            attn_mask = torch.zeros((B, 1, N, M), device=x.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~ctx_key_padding_mask.bool()[:, None, None, :], torch.finfo(q.dtype).min)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)
        return self.proj_out(out.transpose(1, 2).reshape(B, N, C))


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, qkv_bias=True, rope_sattn=None, rope_cattn=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, rope=rope_sattn)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, rope_k=rope_cattn)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio=mlp_ratio)

    def forward(self, x, context, x_pos=None, ctx_pos=None, self_key_padding_mask=None, ctx_key_padding_mask=None):
        x = x + self.self_attn(self.norm1(x), x_pos, key_padding_mask=self_key_padding_mask)
        x = x + self.cross_attn(self.norm2(x), self.norm_context(context), x_pos, ctx_pos, ctx_key_padding_mask=ctx_key_padding_mask)
        x = x + self.ffn(self.norm3(x))
        return x


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class StreamingStateAggregator(nn.Module):
    """
    Compresses (B, T, N, D) patch-wise video embeddings into
    (B, S, state_dim) state tokens via a recurrent dual-decoder.

    chunk_size controls how many tokens are processed per recurrent step.
    For SigLIP-so400m-384 with 27×27=729 patches/frame, set chunk_size=729
    to process one frame per step.
    """

    def __init__(
        self,
        input_dim: int = 1152,
        state_dim: int = 1152,
        num_state_tokens: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        rope_freq: float = 100.0,
        chunk_size: int = 729,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.num_state_tokens = num_state_tokens
        self.num_layers = num_layers
        self.chunk_size = chunk_size

        rope1d = RoPE1D(freq=rope_freq)
        rope2d = RoPE2D(freq=rope_freq)

        self.state_tokens = nn.Embedding(num_state_tokens, input_dim)
        self.state_proj = nn.Linear(input_dim, state_dim)
        self.register_buffer("state_pos", torch.arange(num_state_tokens, dtype=torch.float32))

        self.frame_proj = nn.Linear(input_dim, state_dim)

        self.state_decoder = nn.ModuleList([
            DecoderBlock(state_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         rope_sattn=rope1d, rope_cattn=rope2d)
            for _ in range(num_layers)
        ])
        self.frame_decoder = nn.ModuleList([
            DecoderBlock(state_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         rope_sattn=rope2d, rope_cattn=rope1d)
            for _ in range(num_layers)
        ])
        self.state_norm = nn.LayerNorm(state_dim)
        self.frame_norm = nn.LayerNorm(state_dim)

    @property
    def hidden_size(self) -> int:
        return self.state_dim

    @property
    def config(self) -> dict:
        return {
            "mm_resampler_type": "streaming_agg",
            "mm_streaming_input_dim": self.input_dim,
            "mm_streaming_state_dim": self.state_dim,
            "mm_streaming_num_state_tokens": self.num_state_tokens,
            "mm_streaming_num_layers": self.num_layers,
            "mm_streaming_num_heads": self.state_decoder[0].self_attn.num_heads,
            "mm_streaming_chunk_size": self.chunk_size,
        }

    def _init_state(self, batch_size, device):
        idx = torch.arange(self.num_state_tokens, device=device)
        state = self.state_proj(self.state_tokens(idx))  # (S, state_dim)
        # .repeat() produces a contiguous, properly-tracked tensor; .expand() creates
        # a non-contiguous view that some PyTorch/DeepSpeed versions fail to detect
        # as requiring grad inside torch.utils.checkpoint (use_reentrant=False).
        return state.unsqueeze(0).repeat(batch_size, 1, 1)

    def _build_frame_pos_2d(self, batch_size, chunk_len, device):
        grid_size = math.isqrt(chunk_len)
        if grid_size * grid_size == chunk_len:
            rows = torch.arange(grid_size, device=device, dtype=torch.float32)
            cols = torch.arange(grid_size, device=device, dtype=torch.float32)
            grid = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1).reshape(-1, 2)
        else:
            t = torch.arange(chunk_len, device=device, dtype=torch.float32)
            grid = torch.stack((t, torch.zeros_like(t)), dim=-1)
        return grid.unsqueeze(0).expand(batch_size, -1, -1)

    def _dual_decode(self, state, frame_tokens, state_pos, frame_pos, frame_mask, frame_attn_mask):
        for i, (state_block, frame_block) in enumerate(zip(self.state_decoder, self.frame_decoder)):
            prev_state, prev_frame = state, frame_tokens
            state = state_block(prev_state, prev_frame, state_pos, frame_pos,
                                self_key_padding_mask=None,
                                ctx_key_padding_mask=frame_attn_mask)
            if i < self.num_layers - 1:
                frame_tokens = frame_block(prev_frame, prev_state, frame_pos, state_pos,
                                           self_key_padding_mask=frame_attn_mask,
                                           ctx_key_padding_mask=None)
                frame_tokens = frame_tokens * frame_mask.unsqueeze(-1).to(frame_tokens.dtype)
        return state, frame_tokens

    def forward(
        self,
        frame_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_state_tokens: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            frame_embeddings: (B, T, N, D) patch-wise OR (B, T, D) / (T, D) frame-wise
            mask:             optional (B, T) bool mask, True = valid frame
            return_state_tokens: if True, return (B, S, state_dim) instead of (B, state_dim)

        Returns:
            (B, S, state_dim) if return_state_tokens else (B, state_dim)
            Squeezed to drop the batch dim when input was unbatched.
        """
        frame_embeddings, mask = _flatten_patch_dim(frame_embeddings, mask)

        unbatched = frame_embeddings.ndim == 2
        if unbatched:
            frame_embeddings = frame_embeddings.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        B, T, _ = frame_embeddings.shape
        device = frame_embeddings.device

        if mask is not None:
            mask = mask.bool()

        state = self._init_state(B, device)
        state_pos = self.state_pos.unsqueeze(0).expand(B, -1)

        K = self.chunk_size
        for t0 in range(0, T, K):
            t1 = min(t0 + K, T)
            chunk_len = t1 - t0

            frame_chunk = frame_embeddings[:, t0:t1]
            frame_tokens = self.frame_proj(frame_chunk)  # requires_grad=True from frame_proj weights
            frame_pos = self._build_frame_pos_2d(B, chunk_len, device)

            if mask is not None:
                chunk_mask = mask[:, t0:t1].bool()
            else:
                chunk_mask = torch.ones((B, chunk_len), device=device, dtype=torch.bool)

            active_1d = chunk_mask.any(dim=1)
            chunk_attn_mask = chunk_mask.clone()
            if (~active_1d).any():
                chunk_attn_mask[~active_1d, 0] = True

            frame_tokens = frame_tokens * chunk_mask.unsqueeze(-1).to(frame_tokens.dtype)

            if self.training:
                new_state, _ = torch_checkpoint(
                    self._dual_decode, state, frame_tokens, state_pos,
                    frame_pos, chunk_mask, chunk_attn_mask, use_reentrant=False,
                )
            else:
                new_state, _ = self._dual_decode(
                    state, frame_tokens, state_pos, frame_pos, chunk_mask, chunk_attn_mask,
                )

            state = torch.where(active_1d.view(B, 1, 1), new_state, state)

        if return_state_tokens:
            if unbatched:
                return state.squeeze(0)   # (S, state_dim)
            return state                  # (B, S, state_dim)

        video_embedding = self.state_norm(state).mean(dim=1)
        if unbatched:
            video_embedding = video_embedding.squeeze(0)
        return video_embedding
