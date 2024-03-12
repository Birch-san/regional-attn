import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.utils import USE_PEFT_BACKEND
from torch import FloatTensor
from typing import Optional
from dataclasses import dataclass

from .attn_processor import AttnProcessor
from ..dimensions import Dimensions

# Based on diffusers AttnProcessor2_0 code:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
# - copyright 2023 The HuggingFace Team,
# - Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
# Modifications by Alex Birch to support regional attention
@dataclass
class RegionalAttnProcessor(AttnProcessor):
    expect_size: Dimensions
    
    def __post_init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("RegionalAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: FloatTensor,
        encoder_hidden_states: Optional[FloatTensor] = None,
        attention_mask: Optional[FloatTensor] = None,
        temb: Optional[FloatTensor] = None,
        scale: float = 1.0,
    ) -> FloatTensor:
        assert encoder_hidden_states is not None, "we don't handle self-attention"
        assert attention_mask is None, "we want full management of attn masking"

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key: FloatTensor = attn.to_k(encoder_hidden_states, *args)
        value: FloatTensor = attn.to_v(encoder_hidden_states, *args)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query: FloatTensor = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        out_proj, dropout = attn.to_out

        hidden_states = out_proj(hidden_states, *args)
        hidden_states = dropout(hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states