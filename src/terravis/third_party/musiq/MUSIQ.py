# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# ==============================================================================
# Source Notice
#
# Adapted from pyiqa (IQA-PyTorch) musiq_arch.py: https://github.com/chaofengc/IQA-PyTorch
# Noncommercial use only: PolyForm Noncommercial License 1.0.0,
#   https://polyformproject.org/licenses/noncommercial/1.0.0
# Required Notice: Copyright (c) 2022 Chaofeng Chen
# Architecture and the KonIQ-10k checkpoint (CC BY-NC-SA 4.0) are pyiqa's; only inference is kept.
# pyiqa re-implements google-research/musiq (Apache-2.0, Copyright 2022 The Google Research Authors).
# ==============================================================================

import math
import os
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from itertools import repeat
from typing import Dict, Optional, Tuple, Union
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps


DEFAULT_MUSIQ_REPO_ID = "chaofengc/IQA-PyTorch-Weights"
DEFAULT_MUSIQ_FILENAME = "musiq_koniq_ckpt-e95806b9.pth"


def download_musiq_checkpoint(
    model_weight_root: str,
    repo_id: str = DEFAULT_MUSIQ_REPO_ID,
    filename: str = DEFAULT_MUSIQ_FILENAME,
) -> str:
    """
    Download the fixed MUSIQ checkpoint into model_weight_root.
    """
    os.makedirs(model_weight_root, exist_ok=True)
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=model_weight_root,
    )


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> OrderedDict:
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        cleaned[name] = v
    return cleaned


def load_pretrained_network(
    net: nn.Module,
    model_path: str,
    strict: bool = True,
) -> None:
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = clean_state_dict(state_dict)
    net.load_state_dict(state_dict, strict=strict)


def dist_to_mos(dist_score: torch.Tensor) -> torch.Tensor:
    """
    Convert a score distribution to MOS.
    If the head is scalar regression, just return the scalar value.
    """
    if dist_score.shape[-1] == 1:
        return dist_score
    num_classes = dist_score.shape[-1]
    mos = dist_score * torch.arange(1, num_classes + 1).to(dist_score)
    return mos.sum(dim=-1, keepdim=True)


def _ntuple(n: int):
    def parse(x):
        if isinstance(x, (tuple, list)):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


def symm_pad(im: torch.Tensor, padding: Tuple[int, int, int, int]) -> torch.Tensor:
    """
    Symmetric padding similar to TensorFlow.
    """
    h, w = im.shape[-2:]
    left, right, top, bottom = padding

    x_idx = np.arange(-left, w + right)
    y_idx = np.arange(-top, h + bottom)

    def reflect(x, minx, maxx):
        rng = maxx - minx
        double_rng = 2 * rng
        mod = np.fmod(x - minx, double_rng)
        normed_mod = np.where(mod < 0, mod + double_rng, mod)
        out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
        return np.array(out, dtype=x.dtype)

    x_pad = reflect(x_idx, -0.5, w - 0.5)
    y_pad = reflect(y_idx, -0.5, h - 0.5)
    xx, yy = np.meshgrid(x_pad, y_pad)
    return im[..., yy, xx]


def exact_padding_2d(
    x: torch.Tensor,
    kernel: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    mode: str = "same",
) -> torch.Tensor:
    """
    Exact 2D padding like TensorFlow SAME padding.
    """
    assert len(x.shape) == 4, f"Only support 4D tensor input, but got {x.shape}"

    kernel = to_2tuple(kernel)
    stride = to_2tuple(stride)
    dilation = to_2tuple(dilation)

    _, _, h, w = x.shape

    h2 = math.ceil(h / stride[0])
    w2 = math.ceil(w / stride[1])

    pad_row = (h2 - 1) * stride[0] + (kernel[0] - 1) * dilation[0] + 1 - h
    pad_col = (w2 - 1) * stride[1] + (kernel[1] - 1) * dilation[1] + 1 - w

    pad_l = pad_col // 2
    pad_r = pad_col - pad_l
    pad_t = pad_row // 2
    pad_b = pad_row - pad_t

    mode = mode if mode != "same" else "constant"

    if mode != "symmetric":
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode=mode)
    else:
        x = symm_pad(x, (pad_l, pad_r, pad_t, pad_b))

    return x


class ExactPadding2d(nn.Module):
    def __init__(
        self,
        kernel: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        mode: str = "same",
    ):
        super().__init__()
        self.kernel = to_2tuple(kernel)
        self.stride = to_2tuple(stride)
        self.dilation = to_2tuple(dilation)
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode is None:
            return x
        return exact_padding_2d(x, self.kernel, self.stride, self.dilation, self.mode)


def extract_image_patches(
    x: torch.Tensor,
    kernel: int,
    stride: int = 1,
    dilation: int = 1,
) -> torch.Tensor:
    _, _, h, w = x.shape

    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)

    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w

    x = F.pad(
        x,
        (pad_col // 2, pad_col - pad_col // 2, pad_row // 2, pad_row - pad_row // 2),
    )
    patches = F.unfold(x, kernel, dilation=dilation, stride=stride)
    return patches


def _ceil_divide_int(x: int, y: int) -> int:
    return int(math.ceil(x / y))


def resize_preserve_aspect_ratio(
    image: torch.Tensor,
    h: int,
    w: int,
    longer_side_length: int,
):
    ratio = longer_side_length / max(h, w)
    rh = round(h * ratio)
    rw = round(w * ratio)
    resized = F.interpolate(image, (rh, rw), mode="bicubic", align_corners=False)
    return resized, rh, rw


def _pad_or_cut_to_max_seq_len(x: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    n_crops, c, _ = x.shape
    paddings = torch.zeros((n_crops, c, max_seq_len), device=x.device, dtype=x.dtype)
    x = torch.cat([x, paddings], dim=-1)
    x = x[:, :, :max_seq_len]
    return x


def get_hashed_spatial_pos_emb_index(
    grid_size: int,
    count_h: int,
    count_w: int,
) -> torch.Tensor:
    pos_emb_grid = torch.arange(grid_size).float()

    pos_emb_hash_w = pos_emb_grid.reshape(1, 1, grid_size)
    pos_emb_hash_w = F.interpolate(pos_emb_hash_w, (count_w,), mode="nearest")
    pos_emb_hash_w = pos_emb_hash_w.repeat(1, count_h, 1)

    pos_emb_hash_h = pos_emb_grid.reshape(1, 1, grid_size)
    pos_emb_hash_h = F.interpolate(pos_emb_hash_h, (count_h,), mode="nearest")
    pos_emb_hash_h = pos_emb_hash_h.transpose(1, 2)
    pos_emb_hash_h = pos_emb_hash_h.repeat(1, 1, count_w)

    pos_emb_hash = pos_emb_hash_h * grid_size + pos_emb_hash_w
    pos_emb_hash = pos_emb_hash.reshape(1, -1)
    return pos_emb_hash


def _extract_patches_and_positions_from_image(
    image: torch.Tensor,
    patch_size: int,
    patch_stride: int,
    hse_grid_size: int,
    scale_id: int,
    max_seq_len: int,
) -> torch.Tensor:
    n_crops, c, h, w = image.shape

    patches = extract_image_patches(image, patch_size, patch_stride)
    count_h = _ceil_divide_int(h, patch_stride)
    count_w = _ceil_divide_int(w, patch_stride)

    spatial_p = get_hashed_spatial_pos_emb_index(hse_grid_size, count_h, count_w)
    spatial_p = spatial_p.unsqueeze(1).repeat(n_crops, 1, 1)

    scale_p = torch.ones_like(spatial_p) * scale_id
    mask_p = torch.ones_like(spatial_p)

    out = torch.cat(
        [patches, spatial_p.to(patches), scale_p.to(patches), mask_p.to(patches)],
        dim=1,
    )

    if max_seq_len >= 0:
        out = _pad_or_cut_to_max_seq_len(out, max_seq_len)

    return out


def get_multiscale_patches(
    image: torch.Tensor,
    patch_size: int = 32,
    patch_stride: int = 32,
    hse_grid_size: int = 10,
    longer_side_lengths=(224, 384),
    max_seq_len_from_original_res: Optional[int] = None,
) -> torch.Tensor:
    longer_side_lengths = sorted(longer_side_lengths)

    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    _, _, h, w = image.shape
    outputs = []

    for scale_id, longer_size in enumerate(longer_side_lengths):
        resized_image, _, _ = resize_preserve_aspect_ratio(image, h, w, longer_size)
        max_seq_len = int(np.ceil(longer_size / patch_stride) ** 2)

        out = _extract_patches_and_positions_from_image(
            resized_image,
            patch_size,
            patch_stride,
            hse_grid_size,
            scale_id,
            max_seq_len,
        )
        outputs.append(out)

    if max_seq_len_from_original_res is not None:
        out = _extract_patches_and_positions_from_image(
            image,
            patch_size,
            patch_stride,
            hse_grid_size,
            len(longer_side_lengths),
            max_seq_len_from_original_res,
        )
        outputs.append(out)

    outputs = torch.cat(outputs, dim=-1)
    return outputs.transpose(1, 2)


class StdConv(nn.Conv2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = exact_padding_2d(x, self.kernel_size, self.stride, mode="same")
        weight = self.weight
        weight = weight - weight.mean((1, 2, 3), keepdim=True)
        weight = weight / (weight.std((1, 2, 3), keepdim=True) + 1e-5)
        return F.conv2d(x, weight, self.bias, self.stride)


class Bottleneck(nn.Module):
    def __init__(self, inplanes: int, outplanes: int, stride: int = 1):
        super().__init__()
        width = inplanes

        self.conv1 = StdConv(inplanes, width, 1, 1, bias=False)
        self.gn1 = nn.GroupNorm(32, width, eps=1e-4)

        self.conv2 = StdConv(width, width, 3, 1, bias=False)
        self.gn2 = nn.GroupNorm(32, width, eps=1e-4)

        self.conv3 = StdConv(width, outplanes, 1, 1, bias=False)
        self.gn3 = nn.GroupNorm(32, outplanes, eps=1e-4)

        self.relu = nn.ReLU(True)

        self.needs_projection = inplanes != outplanes or stride != 1
        if self.needs_projection:
            self.conv_proj = StdConv(inplanes, outplanes, 1, stride, bias=False)
            self.gn_proj = nn.GroupNorm(32, outplanes, eps=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.needs_projection:
            identity = self.gn_proj(self.conv_proj(identity))

        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        x = self.gn3(self.conv3(x))
        return self.relu(x + identity)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        bias: bool = False,
        attn_drop: float = 0.0,
        out_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.query = nn.Linear(dim, dim, bias=bias)
        self.key = nn.Linear(dim, dim, bias=bias)
        self.value = nn.Linear(dim, dim, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(out_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, c = x.shape

        q = self.query(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        k = self.key(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        v = self.value(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask_h = mask.reshape(b, 1, n, 1)
            mask_w = mask.reshape(b, 1, 1, n)
            mask2d = mask_h * mask_w
            attn = attn.masked_fill(mask2d == 0, -1e3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.out(x)
        x = self.out_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        num_heads: int,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=1e-6)
        self.attention = MultiHeadAttention(dim, num_heads, bias=True, attn_drop=attn_drop)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor, inputs_masks: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.attention(y, inputs_masks)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AddHashSpatialPositionEmbs(nn.Module):
    def __init__(self, spatial_pos_grid_size: int, dim: int):
        super().__init__()
        self.position_emb = nn.Parameter(
            torch.randn(1, spatial_pos_grid_size * spatial_pos_grid_size, dim)
        )
        nn.init.normal_(self.position_emb, std=0.02)

    def forward(self, inputs: torch.Tensor, inputs_positions: torch.Tensor) -> torch.Tensor:
        return inputs + self.position_emb.squeeze(0)[inputs_positions.long()]


class AddScaleEmbs(nn.Module):
    def __init__(self, num_scales: int, dim: int):
        super().__init__()
        self.scale_emb = nn.Parameter(torch.randn(num_scales, dim))
        nn.init.normal_(self.scale_emb, std=0.02)

    def forward(self, inputs: torch.Tensor, inputs_scale_positions: torch.Tensor) -> torch.Tensor:
        return inputs + self.scale_emb[inputs_scale_positions.long()]


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_dim: int = 1152,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.0,
        num_heads: int = 6,
        num_layers: int = 14,
        num_scales: int = 3,
        spatial_pos_grid_size: int = 10,
        use_scale_emb: bool = True,
    ):
        super().__init__()

        self.use_scale_emb = use_scale_emb
        self.posembed_input = AddHashSpatialPositionEmbs(spatial_pos_grid_size, input_dim)
        self.scaleembed_input = AddScaleEmbs(num_scales, input_dim)

        self.cls = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder_norm = nn.LayerNorm(input_dim, eps=1e-6)

        self.transformer = nn.ModuleDict()
        for i in range(num_layers):
            self.transformer[f"encoderblock_{i}"] = TransformerBlock(
                input_dim,
                mlp_dim,
                num_heads,
                drop=dropout_rate,
                attn_drop=attention_dropout_rate,
            )

    def forward(
        self,
        x: torch.Tensor,
        inputs_spatial_positions: torch.Tensor,
        inputs_scale_positions: torch.Tensor,
        inputs_masks: torch.Tensor,
    ) -> torch.Tensor:
        n, _, _ = x.shape

        x = self.posembed_input(x, inputs_spatial_positions)
        if self.use_scale_emb:
            x = self.scaleembed_input(x, inputs_scale_positions)

        cls_token = self.cls.repeat(n, 1, 1)
        x = torch.cat([cls_token, x], dim=1)

        cls_mask = torch.ones((n, 1), device=inputs_masks.device, dtype=inputs_masks.dtype)
        inputs_mask = torch.cat([cls_mask, inputs_masks], dim=1)

        x = self.dropout(x)

        for _, block in self.transformer.items():
            x = block(x, inputs_mask)

        x = self.encoder_norm(x)
        return x


class MUSIQ(nn.Module):
    """
    Fixed MUSIQ model for the KonIQ checkpoint.
    """

    def __init__(
        self,
        patch_size: int = 32,
        num_class: int = 1,
        hidden_size: int = 384,
        mlp_dim: int = 1152,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.0,
        num_heads: int = 6,
        num_layers: int = 14,
        num_scales: int = 3,
        spatial_pos_grid_size: int = 10,
        use_scale_emb: bool = True,
        pretrained_model_path: Optional[str] = None,
        longer_side_lengths=(224, 384),
        max_seq_len_from_original_res: int = -1,
    ):
        super().__init__()

        resnet_token_dim = 64
        self.patch_size = patch_size

        self.data_preprocess_opts = {
            "patch_size": patch_size,
            "patch_stride": patch_size,
            "hse_grid_size": spatial_pos_grid_size,
            "longer_side_lengths": longer_side_lengths,
            "max_seq_len_from_original_res": max_seq_len_from_original_res,
        }

        self.conv_root = StdConv(3, resnet_token_dim, 7, 2, bias=False)
        self.gn_root = nn.GroupNorm(32, resnet_token_dim, eps=1e-6)

        self.root_pool = nn.Sequential(
            nn.ReLU(True),
            ExactPadding2d(3, 2, mode="same"),
            nn.MaxPool2d(3, 2),
        )

        token_patch_size = patch_size // 4

        self.block1 = Bottleneck(resnet_token_dim, resnet_token_dim * 4)

        self.embedding = nn.Linear(
            resnet_token_dim * 4 * token_patch_size ** 2,
            hidden_size,
        )

        self.transformer_encoder = TransformerEncoder(
            hidden_size,
            mlp_dim,
            attention_dropout_rate,
            dropout_rate,
            num_heads,
            num_layers,
            num_scales,
            spatial_pos_grid_size,
            use_scale_emb,
        )

        self.head = nn.Linear(hidden_size, num_class)

        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            x = (x - 0.5) * 2.0

        x = get_multiscale_patches(x, **self.data_preprocess_opts)

        b, seq_len, dim = x.shape

        inputs_spatial_positions = x[:, :, -3]
        inputs_scale_positions = x[:, :, -2]
        inputs_masks = x[:, :, -1].bool()

        x = x[:, :, :-3]
        x = x.reshape(-1, 3, self.patch_size, self.patch_size)

        x = self.conv_root(x)
        x = self.gn_root(x)
        x = self.root_pool(x)
        x = self.block1(x)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(b, seq_len, -1)

        x = self.embedding(x)
        x = self.transformer_encoder(
            x,
            inputs_spatial_positions,
            inputs_scale_positions,
            inputs_masks,
        )

        q = self.head(x[:, 0])
        mos = dist_to_mos(q)
        return mos


def build_musiq_model(
    model_weight_root: str,
    device: Union[str, torch.device],
    dtype: torch.dtype = torch.float32,
    repo_id: str = DEFAULT_MUSIQ_REPO_ID,
    filename: str = DEFAULT_MUSIQ_FILENAME,
) -> nn.Module:
    """
    Build the fixed MUSIQ model and load pretrained weights.
    """
    ckpt_path = download_musiq_checkpoint(
        model_weight_root=model_weight_root,
        repo_id=repo_id,
        filename=filename,
    )

    model = MUSIQ(pretrained_model_path=ckpt_path)
    model = model.to(device).eval()

    device = torch.device(device)
    if device.type != "cpu":
        model = model.to(dtype=dtype)

    return model


def load_image_as_tensor(
    image: Union[str, Image.Image],
    device: Union[str, torch.device],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Load an RGB image and convert it to a BCHW tensor in [0, 1].
    """
    if isinstance(image, str):
        pil_image = Image.open(image)
    else:
        pil_image = image

    pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
    image_np = np.asarray(pil_image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous().unsqueeze(0)

    device = torch.device(device)
    image_tensor = image_tensor.to(device)
    if device.type != "cpu":
        image_tensor = image_tensor.to(dtype)

    return image_tensor
