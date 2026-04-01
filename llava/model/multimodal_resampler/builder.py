import torch

from .masked_drop import MaskedDrop
from .spatial_pool import SpatialPool
from .perceiver import PerceiverResampler
from .qformer import Qformer


class IdentityMap(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_resampler_type": None}


def build_vision_resampler(model_args, delay_load=False, **kwargs):
    resampler_type = getattr(model_args, "mm_resampler_type", None)
    if resampler_type == "masked_drop":
        return MaskedDrop(model_args)
    elif resampler_type == "spatial_pool":
        return SpatialPool(model_args, **kwargs)
    elif resampler_type == "perceiver":
        return PerceiverResampler(model_args, **kwargs)
    elif resampler_type == "qformer":
        return Qformer(model_args, **kwargs)
    elif resampler_type is None:
        return IdentityMap()
    elif resampler_type == "streaming_agg":
        from .streaming_aggregator import StreamingStateAggregator
        return StreamingStateAggregator(
            input_dim=getattr(model_args, "mm_streaming_input_dim", 1152),
            state_dim=getattr(model_args, "mm_streaming_state_dim", 1152),
            num_state_tokens=getattr(model_args, "mm_streaming_num_state_tokens", 512),
            num_layers=getattr(model_args, "mm_streaming_num_layers", 4),
            num_heads=getattr(model_args, "mm_streaming_num_heads", 8),
            mlp_ratio=getattr(model_args, "mm_streaming_mlp_ratio", 4.0),
            chunk_size=getattr(model_args, "mm_streaming_chunk_size", 729),
        )

    raise ValueError(f"Unknown resampler type: {resampler_type}")
