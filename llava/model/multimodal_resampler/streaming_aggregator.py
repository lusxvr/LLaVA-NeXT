"""
StreamingStateAggregator — bidirectional dual decoder.

Architecture: learned state tokens are updated recurrently via a stack of
dual DecoderBlocks.  Each layer per chunk step:

    state_l = StateBlock(state_{l-1}, frame_{l-1})   # state reads from frame
    frame_l = FrameBlock(frame_{l-1}, state_{l-1})   # frame reads from state

Both blocks use the PREVIOUS layer's outputs as cross-attention context.
The frame branch is discarded after each chunk; only state persists.

StateBlock:  state self-attn (RoPE-1D) → cross-attn to frame (RoPE-1D q, RoPE-2D k) → FFN
FrameBlock:  frame self-attn (RoPE-2D) → cross-attn to state (RoPE-2D q, RoPE-1D k) → FFN

A chunk_norm is applied to state after every sub-chunk update.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_patch_dim(
    frame_embeddings: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
):
    """(B, T, N, D) → (B, T*N, D); expand (B, T) mask to (B, T*N)."""
    if frame_embeddings.ndim == 4:
        B, T, N, D = frame_embeddings.shape
        frame_embeddings = frame_embeddings.reshape(B, T * N, D)
        if mask is not None:
            mask = mask.unsqueeze(-1).expand(B, T, N).reshape(B, T * N)
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
    def __init__(self, dim, num_heads=8, rope=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
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
            attn_mask = attn_mask.masked_fill(
                ~key_padding_mask.bool()[:, None, None, :],
                torch.finfo(q.dtype).min,
            )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class CrossAttention(nn.Module):
    """Cross-attention with separate RoPE for queries (x_pos) and keys (ctx_pos)."""

    def __init__(self, dim, num_heads=8, rope_q=None, rope_k=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.proj_q = nn.Linear(dim, dim, bias=True)
        self.proj_k = nn.Linear(dim, dim, bias=True)
        self.proj_v = nn.Linear(dim, dim, bias=True)
        self.proj_out = nn.Linear(dim, dim, bias=True)
        self.rope_q = rope_q
        self.rope_k = rope_k

    def forward(self, x, context, x_pos=None, ctx_pos=None, ctx_key_padding_mask=None):
        B, N, C = x.shape
        M = context.shape[1]
        q = self.proj_q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.proj_k(context).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.proj_v(context).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        if self.rope_q is not None and x_pos is not None:
            q = self.rope_q(q, x_pos)
        if self.rope_k is not None and ctx_pos is not None:
            k = self.rope_k(k, ctx_pos)
        attn_mask = None
        if ctx_key_padding_mask is not None:
            attn_mask = torch.zeros((B, 1, N, M), device=x.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(
                ~ctx_key_padding_mask.bool()[:, None, None, :],
                torch.finfo(q.dtype).min,
            )
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
    """
    Pre-norm decoder block: self-attn → cross-attn → FFN.

    rope_sattn: applied to both self-attention q/k AND cross-attention queries.
    rope_cattn: applied to cross-attention keys.
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, rope_sattn=None, rope_cattn=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = Attention(dim, num_heads=num_heads, rope=rope_sattn)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, rope_q=rope_sattn, rope_k=rope_cattn)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio=mlp_ratio)

    def forward(self, x, context, x_pos=None, ctx_pos=None,
                self_key_padding_mask=None, ctx_key_padding_mask=None):
        x = x + self.self_attn(self.norm1(x), x_pos, key_padding_mask=self_key_padding_mask)
        x = x + self.cross_attn(
            self.norm2(x),
            self.norm_context(context),
            x_pos=x_pos,
            ctx_pos=ctx_pos,
            ctx_key_padding_mask=ctx_key_padding_mask,
        )
        x = x + self.ffn(self.norm3(x))
        return x


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class StreamingStateAggregator(nn.Module):
    """
    Bidirectional streaming aggregator with dual state/frame decoder.

    S learned state tokens are updated recurrently.  At each chunk, a frame
    decoder branch runs in parallel so the frame tokens are enriched by the
    accumulated state before the state cross-attends to them.

    Each layer per step:
        state ← StateBlock(state, frame)   [state self-attn RoPE-1D, frame keys RoPE-2D]
        frame ← FrameBlock(frame, state)   [frame self-attn RoPE-2D, state keys RoPE-1D]
    Both use the PREVIOUS layer's outputs as context.
    Frame branch is discarded after each chunk.
    chunk_norm is applied to state after every sub-chunk update.
    layer_weights (learned, softmax-normalised) fuse the state outputs from
    all L layers into a single representation before carrying state forward —
    analogous to ELMo/BERT layer weighting, applied per chunk step.
    """

    def __init__(self, model_args):
        super().__init__()
        self.input_dim         = model_args.mm_streaming_input_dim
        self.state_dim         = model_args.mm_streaming_state_dim
        self.num_state_tokens  = model_args.mm_streaming_num_state_tokens
        self.num_layers        = model_args.mm_streaming_num_layers
        self.frames_per_chunk  = model_args.mm_streaming_frames_per_chunk
        self.patches_per_frame = model_args.mm_streaming_patches_per_frame
        self.chunk_size        = self.frames_per_chunk * self.patches_per_frame

        num_heads = model_args.mm_streaming_num_heads
        mlp_ratio = model_args.mm_streaming_mlp_ratio
        rope_freq  = 100.0

        rope1d = RoPE1D(freq=rope_freq)
        rope2d = RoPE2D(freq=rope_freq)

        self.state_tokens = nn.Embedding(self.num_state_tokens, self.state_dim)
        self.register_buffer("state_pos", torch.arange(self.num_state_tokens, dtype=torch.long))

        self.frame_proj = nn.Linear(self.input_dim, self.state_dim)

        # State decoder: state self-attn (RoPE-1D), cross-attn to frame (RoPE-2D keys)
        self.state_decoder = nn.ModuleList([
            DecoderBlock(self.state_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         rope_sattn=rope1d, rope_cattn=rope2d)
            for _ in range(self.num_layers)
        ])
        # Frame decoder: frame self-attn (RoPE-2D), cross-attn to state (RoPE-1D keys)
        self.frame_decoder = nn.ModuleList([
            DecoderBlock(self.state_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         rope_sattn=rope2d, rope_cattn=rope1d)
            for _ in range(self.num_layers)
        ])

        self.layer_weights = nn.Parameter(torch.zeros(self.num_layers))
        self.chunk_norm = nn.LayerNorm(self.state_dim)
        self.state_norm = nn.LayerNorm(self.state_dim)

        self._frame_pos_cache: dict = {}

    # ------------------------------------------------------------------
    # LLaVA builder / trainer interface
    # ------------------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        return self.state_dim

    @property
    def output_dim(self) -> int:
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
            "mm_streaming_mlp_ratio": self.state_decoder[0].ffn.fc1.out_features / self.state_dim,
            "mm_streaming_frames_per_chunk": self.frames_per_chunk,
            "mm_streaming_patches_per_frame": self.patches_per_frame,
        }

    # ------------------------------------------------------------------
    # Recurrent interface
    # ------------------------------------------------------------------

    def init_state(self, batch_size: int, device) -> torch.Tensor:
        """Return (B, S, state_dim) initial recurrent state."""
        idx = torch.arange(self.num_state_tokens, device=device)
        return self.state_tokens(idx).unsqueeze(0).repeat(batch_size, 1, 1)

    def _build_frame_pos_2d(self, batch_size, chunk_len, frame_offset, device):
        """Build (B, chunk_len, 2) positions: dim-0 = absolute frame index, dim-1 = patch index."""
        key = (chunk_len, frame_offset, device)
        if key not in self._frame_pos_cache:
            ppf = self.patches_per_frame
            assert chunk_len % ppf == 0, (
                f"chunk_len ({chunk_len}) is not divisible by patches_per_frame ({ppf}). "
                f"patches_per_frame must match the vision encoder's patch count per frame."
            )
            n_frames = chunk_len // ppf
            t_pos = (
                torch.arange(n_frames, device=device, dtype=torch.float32) + frame_offset
            ).repeat_interleave(ppf)
            s_pos = torch.arange(ppf, device=device, dtype=torch.float32).repeat(n_frames)
            self._frame_pos_cache[key] = torch.stack([t_pos, s_pos], dim=-1)
        return self._frame_pos_cache[key].unsqueeze(0).expand(batch_size, -1, -1)

    def _dual_decode(self, state, frame_embeddings_slice, state_pos, frame_pos, frame_attn_mask):
        """Project and decode one chunk via the bidirectional dual decoder.

        Uses previous-layer outputs as cross-attention context (not sibling's
        current-layer output).  Frame branch update is skipped on the final
        layer since frame tokens are discarded after each chunk.
        """
        frame_tokens = self.frame_proj(frame_embeddings_slice)
        frame_tokens = frame_tokens * frame_attn_mask.unsqueeze(-1).to(frame_tokens.dtype)

        last = self.num_layers - 1
        layer_states = []
        for i, (state_block, frame_block) in enumerate(zip(self.state_decoder, self.frame_decoder)):
            prev_state = state
            prev_frame = frame_tokens
            state = state_block(
                prev_state, prev_frame,
                x_pos=state_pos, ctx_pos=frame_pos,
                ctx_key_padding_mask=frame_attn_mask,
            )
            layer_states.append(state)
            if i < last:
                frame_tokens = frame_block(
                    prev_frame, prev_state,
                    x_pos=frame_pos, ctx_pos=state_pos,
                    self_key_padding_mask=frame_attn_mask,
                )
                frame_tokens = frame_tokens * frame_attn_mask.unsqueeze(-1).to(frame_tokens.dtype)

        weights = F.softmax(self.layer_weights, dim=0)  # (L,)
        return sum(w * s for w, s in zip(weights, layer_states))

    def step(
        self,
        state: torch.Tensor,
        frame_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        frame_offset: int = 0,
    ) -> torch.Tensor:
        """
        Process one chunk of frame embeddings and return the updated state.

        Args:
            state:            (B, S, state_dim) — current recurrent state
            frame_embeddings: (B, T, N, D) patch-wise OR (B, T*N, D) already flat
            mask:             optional bool mask; (B, T) for 4-D input,
                              (B, T*N) for already-flat 3-D input
            frame_offset:     absolute index of the first frame in this chunk

        Returns:
            (B, S, state_dim) updated state
        """
        frame_embeddings, mask = _flatten_patch_dim(frame_embeddings, mask)
        B = state.shape[0]
        T_flat = frame_embeddings.shape[1]
        device = state.device
        state_pos = self.state_pos.unsqueeze(0).expand(B, -1)

        if mask is not None:
            mask = mask.bool()

        K = self.chunk_size
        for t0 in range(0, T_flat, K):
            t1 = min(t0 + K, T_flat)
            chunk_len = t1 - t0

            abs_frame_offset = frame_offset + t0 // self.patches_per_frame
            frame_pos = self._build_frame_pos_2d(B, chunk_len, abs_frame_offset, device)

            if mask is not None:
                chunk_mask = mask[:, t0:t1].bool()
            else:
                chunk_mask = torch.ones((B, chunk_len), device=device, dtype=torch.bool)

            active = chunk_mask.any(dim=1)
            attn_mask = chunk_mask.clone()
            if (~active).any():
                attn_mask[~active, 0] = True

            if self.training:
                new_state = torch_checkpoint(
                    self._dual_decode, state, frame_embeddings[:, t0:t1], state_pos,
                    frame_pos, attn_mask, use_reentrant=False,
                )
            else:
                new_state = self._dual_decode(
                    state, frame_embeddings[:, t0:t1], state_pos, frame_pos, attn_mask
                )

            state = torch.where(active.view(B, 1, 1), new_state, state)
            state = self.chunk_norm(state)

        return state

    def finalize(
        self,
        state: torch.Tensor,
        return_state_tokens: bool = False,
    ) -> torch.Tensor:
        """
        Apply final norm and optionally pool.

        Args:
            state:               (B, S, state_dim)
            return_state_tokens: True → (B, S, state_dim); False → (B, state_dim)
        """
        normed = self.state_norm(state)
        if return_state_tokens:
            return normed
        return normed.mean(dim=1)

    # ------------------------------------------------------------------
    # Convenience: full-sequence forward
    # ------------------------------------------------------------------

    def forward(
        self,
        frame_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_state_tokens: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            frame_embeddings: (B, T, N, D) or (B, T, D) or unbatched (T, D)
            mask:             optional (B, T) bool mask, True = valid frame
            return_state_tokens: True → (B, S, state_dim); False → (B, state_dim)
        """
        frame_embeddings, mask = _flatten_patch_dim(frame_embeddings, mask)

        unbatched = frame_embeddings.ndim == 2
        if unbatched:
            frame_embeddings = frame_embeddings.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        if mask is not None:
            mask = mask.bool()

        B = frame_embeddings.shape[0]
        state = self.init_state(B, frame_embeddings.device)
        state = self.step(state, frame_embeddings, mask)
        out = self.finalize(state, return_state_tokens)

        if unbatched:
            out = out.squeeze(0)
        return out
