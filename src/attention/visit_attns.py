from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, UNetMidBlock2DCrossAttn, Transformer2DModel
from diffusers.models.attention import Attention, BasicTransformerBlock
from typing import Protocol, NamedTuple
from functools import partial

class VisitReceipt(NamedTuple):
  down_blocks_touched: int
  up_blocks_touched: int
  mid_blocks_touched: int

class AttnAcceptor(Protocol):
  @staticmethod
  def __call__(attn: Attention, level: int) -> None: ...

class _BoundAttnAcceptor(Protocol):
  @staticmethod
  def __call__(attn: Attention) -> None: ...

def _visit_t2d(t2d: Transformer2DModel, attn_acceptor: _BoundAttnAcceptor, self_attn: bool, xattn: bool) -> None:
  for tblock in t2d.transformer_blocks:
    assert isinstance(tblock, BasicTransformerBlock)
    if self_attn:
      attn_acceptor(tblock.attn1)
    if xattn:
      attn_acceptor(tblock.attn2)

def visit_attns(unet: UNet2DConditionModel, levels: int, attn_acceptor: AttnAcceptor, self_attn: bool, xattn: bool) -> VisitReceipt:
  """
  counting from outermost level of UNet, how many levels to apply modifications to
  """
  down_blocks_touched = up_blocks_touched = mid_blocks_touched = 0
  for down_block, level in zip(unet.down_blocks, range(levels)):
    if isinstance(down_block, CrossAttnDownBlock2D):
      down_blocks_touched += 1
      for t2d in down_block.attentions:
        _visit_t2d(t2d, partial(attn_acceptor, level=level), self_attn=self_attn, xattn=xattn)
  
  if levels > len(unet.down_blocks):
    if isinstance(unet.mid_block, UNetMidBlock2DCrossAttn):
      mid_blocks_touched += 1
      for t2d in unet.mid_block.attentions:
        # level count might be wrong here (haven't tested this deep), which might have consequences for inferring expected size
        _visit_t2d(t2d, partial(attn_acceptor, level=len(unet.down_blocks)), self_attn=self_attn, xattn=xattn)

  for up_block, level in zip(reversed(unet.up_blocks), range(levels)):
    if isinstance(up_block, CrossAttnUpBlock2D):
      up_blocks_touched += 1
      for t2d in up_block.attentions:
        _visit_t2d(t2d, partial(attn_acceptor, level=level), self_attn=self_attn, xattn=xattn)
  
  return VisitReceipt(
    down_blocks_touched=down_blocks_touched,
    up_blocks_touched=up_blocks_touched,
    mid_blocks_touched=mid_blocks_touched,
  )