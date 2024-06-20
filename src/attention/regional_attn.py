import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.utils import USE_PEFT_BACKEND
from torch import FloatTensor, BoolTensor
from typing import Optional, Literal, List
from dataclasses import dataclass
from einops import rearrange
import torch
import numpy as np
from numpy.typing import NDArray
from enum import Enum

from .attn_processor import AttnProcessor
from ..dimensions import Dimensions

# by oobug
# License: CC BY-SA 4.0
# https://stackoverflow.com/a/75303062/5257399
def even_divide(num: int, div: int) -> List[int]:
    groupSize, remainder = divmod(num, div)
    return [groupSize + (1 if x < remainder else 0) for x in range(div)]

class RegionStrategy(str, Enum):
    HSplit = 'hsplit'
    VSplit = 'vsplit'
    Mask = 'mask'

# Based on diffusers AttnProcessor2_0 code:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
# - copyright 2023 The HuggingFace Team,
# - Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
# Modifications by Alex Birch to support regional attention
@dataclass
class RegionalAttnProcessor(AttnProcessor):
    expect_size: Dimensions
    embs: FloatTensor
    masks: Optional[BoolTensor]
    cfg_enabled: bool

    # True =  emb.repeat_interleave [uncond, uncond, cond, cond]
    # False = emb.repeat            [uncond, cond, uncond, cond]
    unconds_together: bool

    region_strategy: RegionStrategy
    
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
        # we expect encoder_hidden_states currently contains [*unconds, *pool_conds] (i.e. [uncond, pool_cond].repeat_interleave(n_images) for [noise0, noise1].repeat(n_images))
        encoder_hidden_states_orig = encoder_hidden_states
        encoder_hidden_states = encoder_hidden_states_orig.repeat_interleave(self.embs.size(0), dim=-2) # repeat along sequence dim.
        # uncond and pool_cond are now each as long as "our regional conds concatted". we will use mask to keep just one uncond, and we will overwrite pool_cond with our regional conds.
        # NOTE: we assume CFG is in use
        # typically CFG employs 2 text embeddings (cond, uncond),
        # but more general formulations such as multi-cond guidance can use more.
        conds_per_sample = 2
        cond_start_ix = encoder_hidden_states.size(0)//conds_per_sample
        for ix, emb in enumerate(self.embs.unbind()): # replace pool_conds with our regional conds
            if self.unconds_together: # TODO: torch.narrow? or just encoder_hidden_states[cond_start_ix:] = self.embs.flatten(end_dim=1)
                encoder_hidden_states[cond_start_ix:, 77*ix:77*(ix+1), :] = emb
            else: # TODO: encoder_hidden_states[cond_start_ix::conds_per_sample] = self.embs.flatten(end_dim=1)
                encoder_hidden_states[cond_start_ix::conds_per_sample, 77*ix:77*(ix+1), :] = emb

        batch_size, sequence_length, _ = encoder_hidden_states.shape

        attention_mask = hidden_states.new_ones((batch_size, attn.heads, *(self.expect_size), sequence_length), dtype=torch.bool)
        match self.region_strategy:
            case RegionStrategy.HSplit | RegionStrategy.VSplit:
                match self.region_strategy:
                    case RegionStrategy.HSplit:
                        canvas_length = self.expect_size.width
                    case RegionStrategy.VSplit:
                        canvas_length = self.expect_size.height
                pix_splits: List[int] = even_divide(canvas_length, self.embs.size(0))
                pix_lens: NDArray = np.array(pix_splits)
                pix_indices: NDArray = np.roll(np.cumsum(pix_lens),1)
                pix_indices[0] = 0
                if self.unconds_together:
                    # hide the repeated uncond
                    attention_mask[:cond_start_ix, :,:,:,77:] = 0
                    # start by hiding all conds
                    attention_mask[cond_start_ix:,:,:,:,:] = 0
                    for prompt_ix, (pix_start, pix_len) in enumerate(zip(pix_indices, pix_lens)):
                        # reveal just relevant cond
                        match self.region_strategy:
                            case RegionStrategy.HSplit:
                                attention_mask[cond_start_ix:,:,:,pix_start:pix_start+pix_len,77*prompt_ix:77*(prompt_ix+1)] = 1
                            case RegionStrategy.VSplit:
                                attention_mask[cond_start_ix:,:,pix_start:pix_start+pix_len,:,:77*prompt_ix:77*(prompt_ix+1)] = 1
                else:
                    # hide the repeated uncond
                    attention_mask[::conds_per_sample,:,:,:,77:] = 0
                    # start by hiding all conds
                    attention_mask[cond_start_ix::conds_per_sample,:,:,:,:] = 0
                    for prompt_ix, (pix_start, pix_len) in enumerate(zip(pix_indices, pix_lens)):
                        # reveal just relevant cond
                        match self.region_strategy:
                            case RegionStrategy.HSplit:
                                attention_mask[cond_start_ix::conds_per_sample,:,:,pix_start:pix_start+pix_len,77*prompt_ix:77*(prompt_ix+1)] = 1
                            case RegionStrategy.VSplit:
                                attention_mask[cond_start_ix::conds_per_sample,:,pix_start:pix_start+pix_len,:,77*prompt_ix:77*(prompt_ix+1)] = 1
            case RegionStrategy.Mask:
                if self.unconds_together:
                    # hide the repeated uncond
                    attention_mask[:cond_start_ix, :,:,:,77:] = 0
                    # start by hiding all conds
                    attention_mask[cond_start_ix:,:,:,:,:] = 0
                    # attention_mask[cond_start_ix:,:,:,pix_start:pix_start+pix_len,77*prompt_ix:77*(prompt_ix+1)] = 1
                    # attention_mask[cond_start_ix:,:,:,:,:] = self.masks.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    attention_mask[cond_start_ix:].unflatten(-1, (-1, 77)).copy_(rearrange(self.masks, 'b h w -> 1 1 h w b 1').contiguous())
                else:
                    # hide the repeated uncond
                    attention_mask[::conds_per_sample,:,:,:,77:] = 0
                    # start by hiding all conds
                    attention_mask[cond_start_ix::conds_per_sample,:,:,:,:] = 0
                    attention_mask[cond_start_ix::conds_per_sample].unflatten(-1, (-1, 77)).copy_(rearrange(self.masks, 'b h w -> 1 1 h w b 1').contiguous())

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key: FloatTensor = attn.to_k(encoder_hidden_states)
        value: FloatTensor = attn.to_v(encoder_hidden_states)

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
        hidden_states = out_proj(hidden_states)
        hidden_states = dropout(hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states