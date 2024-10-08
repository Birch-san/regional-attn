from contextlib import nullcontext
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput
from transformers import CLIPPreTrainedModel, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPTextModelOutput
import torch
from torch import BoolTensor, FloatTensor, LongTensor, ByteTensor, Generator, inference_mode, cat, randn, tensor, stack
from torchvision.io import read_image
from torchvision.transforms.functional import pil_to_tensor
from torch.nn.functional import pad
from typing import List, Union, Optional, Callable, Dict, Any
from logging import getLogger, Logger
from k_diffusion.sampling import get_sigmas_karras, sample_dpmpp_2m, sample_dpmpp_3m_sde, BrownianTreeNoiseSampler
from os import makedirs, listdir
from os.path import join
import fnmatch
from pathlib import Path
from PIL import Image
from functools import partial
from time import perf_counter

from src.iteration.batched import batched
from src.denoisers.denoiser_proto import Denoiser
from src.denoisers.denoiser_factory import DenoiserFactory, DenoiserFactoryFactory
from src.denoisers.dispatch_denoiser import DispatchDenoiser, IdentifiedDenoiser
from src.denoisers.eps_denoiser import EPSDenoiser
from src.denoisers.cfg_denoiser import CFGDenoiser
from src.device import DeviceType, get_device_type
from src.schedules import KarrasScheduleParams, KarrasScheduleTemplate, get_template_schedule
from src.schedule_params import get_alphas, get_alphas_cumprod, get_betas, quantize_to
from src.latents_shape import LatentsShape
from src.added_cond import CondKwargs
from src.dimensions import Dimensions
from src.time_ids import get_time_ids, get_time_ids_aesthetic
from src.device_ctx import to_device
from src.rgb_to_pil import rgb_to_pil
from src.latents_to_pils import LatentsToBCHW, LatentsToPils, make_latents_to_bchw, make_latents_to_pils
from src.log_intermediates import LogIntermediatesFactory, LogIntermediates, make_log_intermediates_factory
from src.attention.visit_attns import AttnAcceptor, visit_attns
from src.attention.set_attn_processor import set_attn_processor, make_regional_attn

logger: Logger = getLogger(__file__)

device_type: DeviceType = get_device_type()
device = torch.device(device_type)

# https://birchlabs.co.uk/machine-learning#denoise-in-fp16-sample-in-fp32
sampling_dtype = torch.float32

# if you have low VRAM, then we can swap Unets and VAE into VRAM and back as needed.
# if you have high VRAM: don't bother with this because swapping costs time.
# if you're on a Mac: don't bother with this; VRAM and RAM are the same thing.
swap_models = False

use_kohaku = False

stability_model_name = 'stabilityai/stable-diffusion-xl-base-0.9'
kohaku_model_name = 'KBlueLeaf/Kohaku-XL-Delta'
default_model_name = kohaku_model_name if use_kohaku else stability_model_name

use_refiner = False
unets: List[UNet2DConditionModel] = [UNet2DConditionModel.from_pretrained(
  unet_name,
  torch_dtype=torch.float16,
  use_safetensors=unet_use_safetensors,
  variant=unet_variant,
  subfolder='unet',
).eval() for unet_name, unet_variant, unet_use_safetensors in zip([
    default_model_name,
    *(['stabilityai/stable-diffusion-xl-refiner-0.9'] * use_refiner),
  ], [
    'fp16' if default_model_name == stability_model_name else None,
    *([None] * use_refiner),
  ], [
    default_model_name != kohaku_model_name,
    *([True] * use_refiner),
  ])
]
base_unet: UNet2DConditionModel = unets[0]
refiner_unet: Optional[UNet2DConditionModel] = unets[1] if use_refiner else None

compile = False
if compile:
  for unet in unets:
    torch.compile(unet, mode='reduce-overhead', fullgraph=True)

profile = False

tokenizers: List[CLIPTokenizer] = [CLIPTokenizer.from_pretrained(
  default_model_name,
  subfolder=subfolder,
) for subfolder in ('tokenizer', 'tokenizer_2')]
tok_vit_l, tok_vit_big_g = tokenizers

# base uses ViT-L **and** ViT-bigG
# refiner uses just ViT-bigG
vit_l: CLIPTextModel = CLIPTextModel.from_pretrained(
  default_model_name,
  torch_dtype=torch.float16,
  use_safetensors=default_model_name != kohaku_model_name,
  variant='fp16' if default_model_name == stability_model_name else None,
  subfolder='text_encoder',
).eval()

vit_big_g: CLIPTextModelWithProjection = CLIPTextModelWithProjection.from_pretrained(
  default_model_name,
  torch_dtype=torch.float16,
  use_safetensors=default_model_name != kohaku_model_name,
  variant='fp16' if default_model_name == stability_model_name else None,
  subfolder='text_encoder_2',
).eval()

text_encoders: List[CLIPPreTrainedModel] = [vit_l, vit_big_g]

use_ollin_vae = True
vae_kwargs: Dict[str, Any] = {
  'torch_dtype': torch.float16,
} if use_ollin_vae else {
  'variant': 'fp16',
  'subfolder': 'vae',
  # decoder gets NaN result in float16.
  'torch_dtype': torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
}
vae_name = 'madebyollin/sdxl-vae-fp16-fix' if use_ollin_vae else default_model_name

vae: AutoencoderKL = AutoencoderKL.from_pretrained(
  vae_name,
  use_safetensors=True,
  **vae_kwargs,
)
vae.enable_slicing()
# this script won't be using the encoder
del vae.encoder
vae_resample_factor: int = 1 << (len(vae.config.block_out_channels) - 1)
vae_scaling_factor: float = vae.config.scaling_factor
vae_dtype = vae.dtype

# TODO: code-share with how we decode images in the non-intermediates case
latents_to_bchw: LatentsToBCHW = make_latents_to_bchw(vae)
latents_to_pils: LatentsToPils = make_latents_to_pils(latents_to_bchw)
make_log_intermediates: LogIntermediatesFactory = make_log_intermediates_factory(latents_to_pils)
log_intermediates_enabled = False

def round_down(num: int, divisor: int) -> int:
  return num - (num % divisor)

area = 1024**2
# SDXL's aspect-ratio bucketing trained on multiple-of-64 side lengths with close-to-1024**2 area
# width_px: int = round_down(1024, 64)
# width_px: int = round_down(832, 64)
width_px: int = round_down(1216, 64)
# width_px: int = round_down(1280, 64) # don't use 1280, it seems uncommonly low-quality
# width_px: int = round_down(1344, 64)
height_px: int = round_down(area//width_px, 64)
width_lt: int = width_px // vae_resample_factor
height_lt: int = height_px // vae_resample_factor

# TODO: is it height, width or the other way around?
# note: I think these conditions were mainly useful at training time
# (to teach SDXL to distinguish a native-resolution image from an upscaled one).
# I think at inference time you just want "a native-resolution image".
# so I think setting both original and target to (height_px, width_px) makes sense.
# and crop(0,0) (i.e. a well-framed image) make sense.
# TODO: should non-square images condition on orig=targ=(height_px, width_px),
#       or should they use orig=targ=(1024, 1024)?
original_size = Dimensions(height_px, width_px)
target_size = Dimensions(height_px, width_px)
crop_coords_top_left = Dimensions(0, 0)
aesthetic_score = 6.
negative_aesthetic_score = 2.5

cfg_scale = 5.
refiner_cfg_scale = cfg_scale
# https://arxiv.org/abs/2305.08891
# Common Diffusion Noise Schedules and Sample Steps are Flawed
# 3.4. Rescale Classifier-Free Guidance
cfg_rescale = 0.
# cfg_rescale = 0.7

# how was SDXL trained? on zero uncond, or empty string uncond? how about for the refiner?
# https://twitter.com/s_alt_acc/status/1683627077315227648
# https://twitter.com/Birchlabs/status/1683949307450228741
force_zeros_for_empty_prompt = True

uncond_prompt: Optional[str] = None if force_zeros_for_empty_prompt else ''

def read_bitmask(path: str) -> BoolTensor:
  with Image.open(path) as mask_pil:
    m: ByteTensor = pil_to_tensor(mask_pil)
  return m.bool().squeeze(0)

masks: Optional[BoolTensor] = None
# bitmasks: BoolTensor = torch.stack([read_bitmask(path) for path in [
#   'input/blob-bigger-0.png',
#   'input/blob-bigger-1.png',
#   'input/blob-bigger-2.1.png',
# ]]).to(device)
# nor_mask = ~(bitmasks.any(0))
# pool_mask = torch.ones_like(bitmasks[0])
# masks: BoolTensor = torch.cat([
#   # could also do a pool_mask for stuff we want to apply across the board, or just more protection against NaN (i.e. if a certain pixel is not revealed by any mask, which could happen due to bad downsampling)
#   nor_mask.unsqueeze(0),
#   # pool_mask.unsqueeze(0),
#   bitmasks,
# ])

# negative_prompt: Optional[str] = uncond_prompt
# negative_prompt: Optional[str] = ''
# negative_prompt: Optional[str] = 'low quality, blurry, weird proportions, unrealistic, uninteresting, ugly'
negative_prompt: Optional[str] = 'worst quality, low quality, normal quality, old, early, lowres, bad anatomy, blurry, cropped, text, jpeg artifacts, signature, watermark, username, artist name, trademark, title, multiple view, reference sheet, long body, disfigured, ugly, monochrome, transparent background'
# negative_prompt: Optional[str] = 'worst quality, low quality, blurry, lowres'
# pool_prompt = f'digital painting of dragon girl, wings, masterpiece, dramatic, highly detailed, high dynamic range'
# pool_prompt = 'dragon girl, wings, masterpiece, dramatic, highly detailed, high dynamic range'
pool_prompt = 'illustration of dragon girl, wings, masterpiece, dramatic, highly detailed, high dynamic range'
# pool_prompt = 'acacia wood platter, presented on dining table in natural light, ambient occlusion, cinematic, peaceful'
# pool_prompt = 'illustration of dragons fighting in a fantasy battleground'
# nor_prompt = 'illustration of a fantasy battleground'
# pool_prompt = 'empty bamboo wood platter, presented on dining table in natural light, ambient occlusion, cinematic, peaceful'
cond_prompts: List[str] = [
  # nor_prompt,
  # pool_prompt,
  # 'battenberg on acacia wood platter',
  'macaron on acacia wood platter',
  # 'key',
  # 'rock',
  # 'cake',
  # 'macaron on acacia wood platter',
  # 'fire',
  # 'macaron',
  # 'illustration of ice dragon fighting in a fantasy battleground',
  # 'illustration of fire dragon fighting in a fantasy battleground',
  # 'illustration of lightning dragon fighting in a fantasy battleground',
  'illustration of sakura tree in full bloom, petals falling, spring day, calm sky, rolling hills, grass, flowers masterpiece, dramatic, highly detailed, high dynamic range',
  "illustration of ice dragon girl, wings, winter day, masterpiece, dramatic, highly detailed, high dynamic range, watercolor (medium)",
  'illustration of fire dragon girl, wings, night, masterpiece, dramatic, highly detailed, high dynamic range, aurora borealis',
]
prompts: List[str] = [
  *([] if negative_prompt is None or cfg_scale == 1. else [negative_prompt]),
  pool_prompt,
  *cond_prompts,
]

tokenizeds: List[BatchEncoding] = [tokenizer(
  prompts,
  padding="max_length",
  max_length=tokenizer.model_max_length,
  truncation=True,
  return_overflowing_tokens=True,
) for tokenizer in tokenizers]
enc_vit_l, enc_vit_big_g = tokenizeds

# we expect to get one embedding (of penultimate hidden states) per text encoder
embeddings: List[FloatTensor] = []
embedding_masks: List[BoolTensor] = []
# we expect to get just one pooled embedding, from vit_big_g
pooled_embed: Optional[FloatTensor] = None
for tokenizer_ix, (tokenized, tokenizer, text_encoder) in enumerate(zip(tokenizeds, tokenizers, text_encoders)):
  overflows: List[List[int]] = tokenized.data['overflowing_tokens']
  overflows_decoded: List[str] = tokenizer.batch_decode(overflows)
  num_truncateds: List[int] = tokenized.data['num_truncated_tokens']
  for prompt_ix, (overflow_decoded, num_truncated) in enumerate(zip(overflows_decoded, num_truncateds)):
    if num_truncated > 0:
      logger.warning(f"Prompt {prompt_ix} will be truncated, due to exceeding tokenizer {tokenizer_ix}'s length limit by {num_truncated} tokens. Overflowing portion of text was: <{overflow_decoded}>")

  input_ids: LongTensor = tensor(tokenized.input_ids, device=device)
  attention_mask: BoolTensor = tensor(tokenized.attention_mask, device=device, dtype=torch.bool)
  embedding_masks.append(attention_mask)

  with to_device(text_encoder, device), inference_mode():
    assert isinstance(text_encoder, CLIPTextModel) or isinstance(text_encoder, CLIPTextModelWithProjection), f'text_encoder of type "{type(text_encoder)}" was unexpectedly neither a CLIPTextModel nor a CLIPTextModelWithProjection'
    encoder_out: Union[BaseModelOutputWithPooling, CLIPTextModelOutput] = text_encoder(
      input_ids=input_ids,
      # TODO: check what mask is like. I dunno whether these CLIPs were trained with masked input, and dunno whether Stability
      #       inferenced from these CLIPs with a mask, but my guess for the latter is not.
      # attention_mask=attention_mask,
      output_hidden_states=True,
      return_dict=True,
    )
    embedding: FloatTensor = encoder_out.hidden_states[-2]
    if isinstance(encoder_out, CLIPTextModelOutput): # returned by CLIPTextModelWithProjection, e.g. vit_big_g
      assert pooled_embed is None, 'we only expected one of the text encoders (vit_big_g) to be a CLIPTextModelWithProjection capable of providing a pooled embedding.'
      pooled_embed: FloatTensor = encoder_out.text_embeds
  embeddings.append(embedding)

assert pooled_embed is not None
base_embed: FloatTensor = cat(embeddings, dim=-1)
base_embedding_mask: BoolTensor = cat(embedding_masks, dim=-1)
if use_refiner:
  refiner_embed: FloatTensor = embeddings[1] # emb_vit_big_g
  refiner_embedding_mask: BoolTensor = embedding_masks[1] # mask_vit_big_g

if negative_prompt is None and cfg_scale > 1.:
  base_embed = pad(base_embed, pad=(0,0, 0,0, 1,0), mode='constant')
  pooled_embed = pad(pooled_embed, pad=(0,0, 1,0), mode='constant')
  base_embedding_mask = pad(base_embedding_mask, pad=(0,0, 1,0), mode='constant', value=True)

  if use_refiner:
    refiner_embed = pad(refiner_embed, pad=(0,0, 0,0, 1,0), mode='constant')

# we separate pooled embeds (neg cond, pool cond)
# from region-specific embeds
regional_cond_start_ix = 1 if negative_prompt is None or cfg_scale == 1. else 2
pooled_embed = pooled_embed[:regional_cond_start_ix]
# note: though we make the pool emb available for xattn to see, we may choose to ignore it in favour of regional conds
base_cond_emb: FloatTensor = base_embed[regional_cond_start_ix:]
base_embed = base_embed[:regional_cond_start_ix]
if use_refiner:
  refiner_cond_emb: FloatTensor = refiner_embed[regional_cond_start_ix:]
  refiner_embed = refiner_embed[:regional_cond_start_ix]

base_time_ids: FloatTensor = get_time_ids(
  original_size=original_size,
  crop_coords_top_left=crop_coords_top_left,
  target_size=target_size,
  dtype=base_embed.dtype,
  device=device,
)
refiner_time_ids: FloatTensor = cat([
  get_time_ids_aesthetic(
    original_size=original_size,
    crop_coords_top_left=crop_coords_top_left,
    aesthetic_score=score,
    dtype=base_embed.dtype,
    device='cpu',
  ) for score in [*([negative_aesthetic_score] if cfg_scale > 1. else []), aesthetic_score]
]).to(device)

alphas_cumprod: FloatTensor = get_alphas_cumprod(get_alphas(get_betas(device=device))).to(dtype=sampling_dtype)

unets_k_wrapped: List[EPSDenoiser] = [
  EPSDenoiser(
    unet,
    alphas_cumprod,
    sampling_dtype,
  ) for unet in [base_unet, *([refiner_unet] * use_refiner)]
]
base_unet_k_wrapped: EPSDenoiser = unets_k_wrapped[0]
refiner_unet_k_wrapped: Optional[EPSDenoiser] = unets_k_wrapped[1] if use_refiner else None

def make_cfg_denoiser(
  delegate: Denoiser,
  cross_attention_conds: FloatTensor,
  added_cond_kwargs: CondKwargs,
  cfg_scale: float,
  cross_attention_mask: Optional[BoolTensor] = None,
) -> CFGDenoiser:
  return CFGDenoiser(
    denoiser=delegate,
    cross_attention_conds=cross_attention_conds,
    added_cond_kwargs=added_cond_kwargs,
    cfg_scale=cfg_scale,
    guidance_rescale=cfg_rescale,
    cross_attention_mask=cross_attention_mask,
  )
denoiser_factory_factory: DenoiserFactoryFactory[CFGDenoiser] = lambda delegate: partial(make_cfg_denoiser, delegate)

base_denoiser_factory: DenoiserFactory[Denoiser] = denoiser_factory_factory(base_unet_k_wrapped)
refiner_denoiser_factory: Optional[DenoiserFactory[Denoiser]] = denoiser_factory_factory(refiner_unet_k_wrapped) if use_refiner else None

# schedule_template = KarrasScheduleTemplate.CudaMasteringMaximizeRefiner
schedule_template = KarrasScheduleTemplate.CudaMastering
# schedule_template = KarrasScheduleTemplate.Searching
# schedule_template = KarrasScheduleTemplate.Mastering32
# schedule_template = KarrasScheduleTemplate.Mastering
schedule: KarrasScheduleParams = get_template_schedule(
  schedule_template,
  model_sigma_min=base_unet_k_wrapped.sigma_min,
  model_sigma_max=base_unet_k_wrapped.sigma_max,
  device=base_unet_k_wrapped.sigmas.device,
  dtype=base_unet_k_wrapped.sigmas.dtype,
)

steps, sigma_max, sigma_min, rho = schedule.steps, schedule.sigma_max, schedule.sigma_min, schedule.rho
sigmas: FloatTensor = get_sigmas_karras(
  n=steps,
  sigma_max=sigma_max.cpu(),
  sigma_min=sigma_min.cpu(),
  rho=rho,
  device=device,
).to(sampling_dtype)

print(f"sigmas (unquantized):\n{', '.join(['%.4f' % s.item() for s in sigmas])}")
sigmas_quantized = pad(quantize_to(sigmas[:-1], base_unet_k_wrapped.sigmas), pad=(0, 1))
print(f"sigmas (quantized):\n{', '.join(['%.4f' % s.item() for s in sigmas_quantized])}")
sigmas = sigmas_quantized

if use_refiner:
  # (SDXL technical report, 2.5)
  # refiner specializes in the final 200 timesteps of the denoising schedule,
  # i.e. sigmas starting from:
  #   0.5692849159240723
  # I assume this corresponds to img2img strength of 0.2
  refine_from_sigma: float = base_unet_k_wrapped.sigmas[199].item()
  print(f"the highest sigma within the refiner's training distribution is: {refine_from_sigma:.4f} ")

  base_purview: BoolTensor = sigmas>refine_from_sigma
  refiner_purview: BoolTensor = sigmas<=refine_from_sigma
  base_duty: int = base_purview.sum()
  refiner_duty: int = refiner_purview[:-1].sum() # zero doesn't count
  print(f"base model will be active during {base_duty} sigmas:\n{', '.join(['%.4f' % s.item() for s in sigmas[base_purview]])}")
  print(f"refiner model will be active during {refiner_duty} sigmas:\n{', '.join(['%.4f' % s.item() for s in sigmas[refiner_purview][:-1]])}")

# note: if you ever change this script into img2img, then you will want to start the
# denoising from a later sigma than sigma_max.
start_sigma = sigmas[0]

out_dir = 'out'
intermediates_path=f'{out_dir}/intermediates'
for path_ in [out_dir, intermediates_path]:
  makedirs(path_, exist_ok=True)

out_imgs_unsorted: List[str] = fnmatch.filter(listdir(out_dir), f'*_*.*')
get_out_ix: Callable[[str], int] = lambda stem: int(stem.split('_', maxsplit=1)[0])
out_keyer: Callable[[str], int] = lambda fname: get_out_ix(Path(fname).stem)
out_imgs: List[str] = sorted(out_imgs_unsorted, key=out_keyer)
next_ix = get_out_ix(Path(out_imgs[-1]).stem)+1 if out_imgs else 0

latents_shape = LatentsShape(base_unet.config.in_channels, height_lt, width_lt)

modify_xattn = True
if modify_xattn:
  sample_size = Dimensions(height=latents_shape.height, width=latents_shape.width)
  refiner_cond_embs: List[FloatTensor] = [refiner_cond_emb] if use_refiner else []
  cond_embs: List[FloatTensor] = [base_cond_emb, *refiner_cond_embs]
  for unet, cond_emb in zip(unets, cond_embs):
    regional_attn_maker: AttnAcceptor = partial(
      make_regional_attn,
      sample_size=sample_size,
      embs=cond_emb,
      masks=masks,
      cfg_enabled=True,
      unconds_together=True,
      region_strategy='hsplit',
      # region_strategy='mask',
    )
    attn_setter: AttnAcceptor = partial(set_attn_processor, get_attn_processor=regional_attn_maker)
    visit_receipt = visit_attns(unet, levels=len(unet.down_blocks)+1, attn_acceptor=attn_setter, self_attn=False, xattn=True)
    print(f'Visited attention in {visit_receipt.down_blocks_touched} down blocks, {visit_receipt.up_blocks_touched} up blocks, and {visit_receipt.mid_blocks_touched} mid blocks.')

# we generate with CPU random so that results can be reproduced across platforms
generator = Generator(device='cpu')

max_batch_size = 16

# start_seed = 29
# start_seed = 30
start_seed = 36
# start_seed = 69
# start_seed = 160
sample_count = 1
seeds = range(start_seed, start_seed+sample_count)

if not swap_models:
  base_unet.to(device)
  vae.to(device)
  if use_refiner:
    refiner_unet.to(device)

img_provenance: str = 'refined' if use_refiner else 'base'

print(f'Generating {sample_count} images, in batches of {max_batch_size}')

for batch_ix, batch_seeds in enumerate(batched(seeds, max_batch_size)):
  batch_size: int = len(batch_seeds)
  print(f'Generating a batch of {batch_size} images, seeds {batch_seeds}')
  latents: FloatTensor = stack([
    randn(
      (
        latents_shape.channels,
        latents_shape.height,
        latents_shape.width
      ),
      dtype=sampling_dtype,
      device=generator.device,
      generator=generator.manual_seed(seed),
    ) for seed in batch_seeds
  ]).to(device)
  latents *= start_sigma

  out_stems: List[str] = [
    f'{(next_ix + batch_ix*max_batch_size + sample_ix):05d}_{img_provenance}_{pool_prompt.split(",")[0]}_{seed}'
    for sample_ix, seed in enumerate(batch_seeds)
  ]

  if log_intermediates_enabled:
    intermediates_paths: List[str] = [f'{intermediates_path}/{stem}' for stem in out_stems]
    for intermediates_path in intermediates_paths:
      makedirs(intermediates_path, exist_ok=True)
    callback: LogIntermediates = make_log_intermediates(intermediates_paths)
  else:
    callback = None

  base_added_cond_kwargs = CondKwargs(
    text_embeds=pooled_embed.repeat_interleave(batch_size, 0),
    time_ids=base_time_ids.expand(base_embed.size(0) * batch_size, -1),
  )
  base_denoiser: Denoiser = base_denoiser_factory(
    cross_attention_conds=base_embed.repeat_interleave(batch_size, 0),
    added_cond_kwargs=base_added_cond_kwargs,
    cfg_scale=cfg_scale,
    # TODO: check what mask is like. I assume Stability never trained on masked embeddings.
    # cross_attention_mask=base_embedding_mask.repeat(batch_size, 1),
  )
  if use_refiner:
    refiner_added_cond_kwargs = CondKwargs(
      text_embeds=pooled_embed.repeat_interleave(batch_size, 0),
      time_ids=refiner_time_ids.repeat_interleave(batch_size, 0) if cfg_scale > 1. else refiner_time_ids.expand(base_embed.size(0) * batch_size, -1),
    )
    refiner_denoiser: Denoiser = refiner_denoiser_factory(
      cross_attention_conds=refiner_embed.repeat_interleave(batch_size, 0),
      added_cond_kwargs=refiner_added_cond_kwargs,
      cfg_scale=refiner_cfg_scale,
      # TODO: check what mask is like. I assume Stability never trained on masked embeddings.
      # cross_attention_mask=refiner_embedding_mask.repeat(batch_size, 1),
    )

    base_id = IdentifiedDenoiser('base', base_denoiser)
    refiner_id = IdentifiedDenoiser('refiner', refiner_denoiser)
    def pick_delegate(sigma: float) -> IdentifiedDenoiser:
      return base_id if sigma > refine_from_sigma else refiner_id
    def on_delegate_change(from_: Optional[str], to: str) -> None:
      if from_ == 'base':
        base_unet.cpu()
      if to == 'base':
        base_unet.to(device)
      elif to == 'refiner':
        refiner_unet.to(device)

    denoiser = DispatchDenoiser(
      pick_delegate=pick_delegate,
      on_delegate_change=on_delegate_change if swap_models else None,
    )
  else:
    denoiser: Denoiser = base_denoiser

  noise_sampler = BrownianTreeNoiseSampler(
    latents,
    sigma_min=sigma_min,
    sigma_max=start_sigma,
    seed=batch_seeds,
    # perhaps we should have the dispatch denoiser send the sigma_to_t to the relevant denoiser for the given sigma,
    # but since ensembling relies on the denoisers' agreeing to the same time schedule, we can probably ignore that detail.
    transform=lambda sigma: base_unet_k_wrapped.sigma_to_t(sigma),
  )

  tic: float = perf_counter()

  with inference_mode(), to_device(base_unet, device) if swap_models else nullcontext():
    # denoised_latents: FloatTensor = sample_dpmpp_2m(
    denoised_latents: FloatTensor = sample_dpmpp_3m_sde(
      denoiser,
      latents,
      sigmas,
      callback=callback,
      noise_sampler=noise_sampler,
      # eta=0.95,
    ).to(vae_dtype)

  if profile:
    batch_secs: float = perf_counter() - tic
    per_sample_secs: float = batch_secs/batch_size
    torch.cuda.synchronize()
    print(f"batch-of-{batch_size} took {batch_secs:.2f}s with use_refiner={use_refiner}. that's {per_sample_secs:.2f}s/sample. Measurement includes UNet(s) only.")

  if swap_models:
    refiner_unet.cpu()

  denoised_latents = denoised_latents / vae_scaling_factor

  with inference_mode(), to_device(vae, device) if swap_models else nullcontext():
    decoder_out: DecoderOutput = vae.decode(denoised_latents.to(vae_dtype))
  sample: FloatTensor = decoder_out.sample

  sample: FloatTensor = sample.div(2).add(.5).clamp(0,1)
  for ix, (decoded, out_stem) in enumerate(zip(sample, out_stems)):
    # if you want lossless images: consider png
    out_name: str = join(out_dir, f'{out_stem}.jpg')
    img: Image = rgb_to_pil(decoded)
    img.save(out_name, subsampling=0, quality=95)
    print(f'Saved image: {out_name}')