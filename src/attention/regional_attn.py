import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.utils import USE_PEFT_BACKEND
from torch import FloatTensor
from typing import Optional
from dataclasses import dataclass
from einops import rearrange
import torch

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
    embs: FloatTensor
    cfg_enabled: bool

    # True =  emb.repeat_interleave [uncond, uncond, cond, cond]
    # False = emb.repeat            [uncond, cond, uncond, cond]
    unconds_together: bool
    
    def __post_init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("RegionalAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        assert self.cfg_enabled, "we only support CFG mode for now"

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
        assert hidden_states.ndim == 3, f"Expected a disappointing 3D tensor that I would have the fun job of unflattening. Instead received {hidden_states.ndim}-dimensional tensor."
        assert hidden_states.size(-2) == self.expect_size.height * self.expect_size.width, "Sequence dimension is not equal to the product of expected height and width, so we cannot unflatten sequence into 2D sequence."

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        encoder_hidden_states_orig = encoder_hidden_states
        encoder_hidden_states = encoder_hidden_states_orig.repeat_interleave(self.embs.size(0), dim=-2)

        # NOTE: we assume CFG is in use
        # typically CFG employs 2 text embeddings (cond, uncond),
        # but more general formulations such as multi-cond guidance can use more.
        conds_per_sample = 2
        cond_start_ix = encoder_hidden_states.size(0)//conds_per_sample
        for ix, emb in enumerate(self.embs.unbind()):
            # unconds together:
            if self.unconds_together:
                encoder_hidden_states[cond_start_ix:, 77*ix:77*(ix+1), :] = emb
            else:
                encoder_hidden_states[cond_start_ix::conds_per_sample, 77*ix:77*(ix+1), :] = emb

        batch_size, sequence_length, _ = encoder_hidden_states.shape

        attention_mask = hidden_states.new_ones((batch_size, attn.heads, *(self.expect_size), sequence_length), dtype=torch.bool)
        # NOTE: we assume CFG is in use
        # for unconds, we don't wish for the uncond to be repeated
        if self.unconds_together:
            attention_mask[:cond_start_ix, :,:,:,77:] = 0
            attention_mask[ cond_start_ix:,:,:,:self.expect_size.width//2, :77] = 0
            attention_mask[ cond_start_ix:,:,:, self.expect_size.width//2:, 77:] = 0
        else:
            attention_mask[::conds_per_sample,:,:,:,77:] = 0
            attention_mask[cond_start_ix::conds_per_sample,:,:,:self.expect_size.width//2, :77] = 0
            attention_mask[cond_start_ix::conds_per_sample,:,:, self.expect_size.width//2:, 77:] = 0

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key: FloatTensor = attn.to_k(encoder_hidden_states, *args)
        value: FloatTensor = attn.to_v(encoder_hidden_states, *args)

        query: FloatTensor = rearrange(query, "n (h w) (nh e) -> n nh h w e", nh=attn.heads, h=self.expect_size.height, w=self.expect_size.width)
        key, value = [rearrange(p, "n s (nh e) -> n nh 1 s e", nh=attn.heads) for p in (key, value)]

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = rearrange(hidden_states, '... nh h w c -> ... (h w) (nh c)')

        out_proj, dropout = attn.to_out
        hidden_states = out_proj(hidden_states, *args)
        hidden_states = dropout(hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states