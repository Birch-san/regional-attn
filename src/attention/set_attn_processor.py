from diffusers.models.attention import Attention
from typing import Protocol, Literal, Optional
from torch import FloatTensor, BoolTensor
from torch.nn.functional import interpolate
import torch
from .attn_processor import AttnProcessor
from .regional_attn import RegionalAttnProcessor
from ..dimensions import Dimensions

class GetAttnProcessor(Protocol):
  @staticmethod
  def __call__(self, attn: Attention, level: int) -> AttnProcessor: ...

def set_attn_processor(attn: Attention, level: int, get_attn_processor: GetAttnProcessor) -> None:
  attn_processor: AttnProcessor = get_attn_processor(attn, level)
  attn.set_processor(attn_processor)

def make_regional_attn(
  attn: Attention,
  level: int,
  sample_size: Dimensions,
  embs: FloatTensor,
  masks: Optional[BoolTensor],
  cfg_enabled: bool,
  unconds_together: bool,
  region_strategy: Literal['hsplit', 'vsplit'],
) -> RegionalAttnProcessor:
  downsampled_size = sample_size
  # yes I know about raising 2 to the power of negative number, but I want to model a repeated rounding-down
  downsample = torch.nn.Conv2d(1,1, kernel_size=3, stride=2, padding=1)
  size_probe = torch.ones(1, sample_size.height, sample_size.width)
  for _ in range(level):
    # haven't actually tested this for levels>1 so I've never run this code
    size_probe = downsample(size_probe)
    height, width = size_probe.shape[1:]
    downsampled_size = Dimensions(height=height, width=width)
  masks = None if masks is None else interpolate(masks.unsqueeze(1).half(), size=downsampled_size, mode='nearest').bool().squeeze(1)
  attn = RegionalAttnProcessor(
    expect_size=downsampled_size,
    embs=embs,
    masks=masks,
    cfg_enabled=cfg_enabled,
    unconds_together=unconds_together,
    region_strategy=region_strategy,
  )
  return attn