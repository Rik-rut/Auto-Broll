# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
##### Enjoy this spagheti VRAM optimizations done by DeepBeepMeep !
# I am sure you are a nice person and as you copy this code, you will give me officially proper credits:
# Please link to https://github.com/deepbeepmeep/Wan2GP and @deepbeepmeep on twitter  

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.normalization import RMSNorm

from broll_agent.shared.attention import pay_attention


def apply_rotary_emb_inplace(x_list: list, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_in = x_list.pop()
    dtype = x_in.dtype
    x = x_in.float().reshape(*x_in.shape[:-1], -1, 2)
    x_in = None
    cos = freqs_cis[..., 0].unsqueeze(2)
    sin = freqs_cis[..., 1].unsqueeze(2)
    x0, x1 = x[..., 0], x[..., 1]
    x0_orig = x0.clone()
    x0.mul_(cos).addcmul_(x1, sin, value=-1)
    x1.mul_(cos).addcmul_(x0_orig, sin)
    return x.flatten(3).to(dtype)


ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size,
                mid_size,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                mid_size,
                out_size,
                bias=True,
            ),
        )

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, qk_norm: bool):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False))

        self.norm_q = RMSNorm(self.head_dim, eps=1e-5) if qk_norm else None
        self.norm_k = RMSNorm(self.head_dim, eps=1e-5) if qk_norm else None

    def forward(self, h_list: list, freqs_cis: torch.Tensor) -> torch.Tensor:
        h = h_list.pop()
        query = self.to_q(h)
        key = self.to_k(h)
        value = self.to_v(h); del h
        query = query.unflatten(-1, (self.n_heads, -1))
        key = key.unflatten(-1, (self.n_heads, -1))
        value = value.unflatten(-1, (self.n_heads, -1))
        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)
        if freqs_cis is not None:
            q_list = [query]; del query
            query = apply_rotary_emb_inplace(q_list, freqs_cis)
            k_list = [key]; del key
            key = apply_rotary_emb_inplace(k_list, freqs_cis)
        dtype = query.dtype

        x_list = [query, key, value]
        del query, key, value
        out = pay_attention(x_list).flatten(2, 3).to(dtype)
        return self.to_out(out)


class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.attention = Attention(dim, n_heads, qk_norm)

        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.ffn_mult = 8 / 3
        self.layer_id = layer_id

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True),
            )

    def _apply_ffn_chunked(self, ffn_in: torch.Tensor) -> None:
        _, seq_len, dim = ffn_in.shape
        ffn_in_flat = ffn_in.reshape(-1, dim)
        chunk_size = max(int(seq_len // self.ffn_mult), 1)
        for ffn_chunk in torch.split(ffn_in_flat, chunk_size):
            ffn_chunk[...] = self.feed_forward(ffn_chunk)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: torch.Tensor | None = None,
    ):
        if self.modulation:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            scale_msa.add_(1.0)
            normed = self.attention_norm1(x)
            normed.mul_(scale_msa)
            attn_out = self.attention_norm2(self.attention([normed], freqs_cis))
            attn_out.mul_(gate_msa.tanh_())
            x.add_(attn_out); attn_out = None
            scale_mlp.add_(1.0)
            normed = self.ffn_norm1(x)
            normed.mul_(scale_mlp)
            self._apply_ffn_chunked(normed)
            ffn_out = self.ffn_norm2(normed); normed = None
            ffn_out.mul_(gate_mlp.tanh_())
            x.add_(ffn_out); ffn_out = None
        else:
            x.add_(self.attention_norm2(self.attention([self.attention_norm1(x)], freqs_cis)))
            normed = self.ffn_norm1(x)
            self._apply_ffn_chunked(normed)
            x.add_(self.ffn_norm2(normed)); normed = None
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = self.adaLN_modulation(c)
        scale.add_(1.0)
        x = self.norm_final(x)
        x.mul_(scale.unsqueeze(1))
        return self.linear(x)


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: list[int] = (16, 56, 56),
        axes_lens: list[int] = (64, 128, 128),
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens), "axes_dims and axes_lens must have the same length"
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: list[int], end: list[int], theta: float = 256.0):
        with torch.device("cpu"):
            freqs_cis = []
            for d, e in zip(dim, end):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
                timestep = torch.arange(e, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis.append(torch.stack([freqs.cos(), freqs.sin()], dim=-1))
            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        else:
            if self.freqs_cis[0].device != device:
                self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=-2)


class ZImageTransformer2DModel(nn.Module):
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        siglip_feat_dim=None,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads

        self.rope_theta = rope_theta
        self.t_scale = t_scale

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

            final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)

        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
                for layer_id in range(n_layers)
            ]
        )

        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

    def unpatchify(self, x: list[torch.Tensor], size: list[tuple], patch_size, f_patch_size) -> list[torch.Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz
        x_out_list = []
        for i in range(bsz):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)
            x_out_list.append( 
                x[i][:ori_len]
                .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.out_channels, F, H, W)
            )
        return torch.stack(x_out_list)

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)

        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def patchify(
        self,
        all_image: list[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
        cap_padding_len: int,
    ):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []

        for i, image in enumerate(all_image):
            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padding_pos_ids = (
                self.create_coordinate_grid(
                    size=(1, 1, 1),
                    start=(0, 0, 0),
                    device=device,
                )
                .flatten(0, 2)
                .repeat(image_padding_len, 1)
            )
            image_padded_pos_ids = torch.cat([image_ori_pos_ids, image_padding_pos_ids], dim=0)
            all_image_pos_ids.append(image_padded_pos_ids)
            all_image_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            image_padded_feat = torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0)
            all_image_out.append(image_padded_feat)

        return (
            all_image_out,
            all_image_size,
            all_image_pos_ids,
            all_image_pad_mask,
        )

    def patchify_and_embed(
        self,
        all_image: list[torch.Tensor],
        all_cap_feats: list[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
    ):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []
        all_cap_pos_ids = []
        all_cap_pad_mask = []
        all_cap_feats_out = []

        for i, (image, cap_feat) in enumerate(zip(all_image, all_cap_feats)):
            cap_ori_len = len(cap_feat)
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)
            all_cap_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((cap_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((cap_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            cap_padded_feat = torch.cat(
                [cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)],
                dim=0,
            )
            all_cap_feats_out.append(cap_padded_feat)

            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_ori_len + cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padding_pos_ids = (
                self.create_coordinate_grid(
                    size=(1, 1, 1),
                    start=(0, 0, 0),
                    device=device,
                )
                .flatten(0, 2)
                .repeat(image_padding_len, 1)
            )
            image_padded_pos_ids = torch.cat([image_ori_pos_ids, image_padding_pos_ids], dim=0)
            all_image_pos_ids.append(image_padded_pos_ids)
            all_image_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            image_padded_feat = torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0)
            all_image_out.append(image_padded_feat)

        return (
            all_image_out,
            all_cap_feats_out,
            all_image_size,
            all_image_pos_ids,
            all_cap_pos_ids,
            all_image_pad_mask,
            all_cap_pad_mask,
        )

    def forward(
        self,
        x_list: list[torch.Tensor],
        t,
        cap_feats_list: list[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
        callback=None,
        pipeline=None,
    ):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        num_noise_samples = len(x_list)
        assert len(cap_feats_list) == num_noise_samples, "cap_feats_list must match x_list length"

        device = x_list[0].device
        t_high = t.to(dtype=torch.float32)
        t_emb = self.t_embedder(t_high.abs() * self.t_scale)
        t = t_emb

        x_embedder = self.all_x_embedder[f"{patch_size}-{f_patch_size}"]
        embedded_x_list = []
        cap_embedded_list = []
        per_sample_kwargs = []

        for i, (x, cap_feats) in enumerate(zip(x_list, cap_feats_list)):
            bsz = x.shape[0]
            (x_patches, cap_out, x_i_size, x_pos_ids, cap_pos_ids, x_inner_pad_mask, cap_inner_pad_mask) = self.patchify_and_embed(x, [cap_feats] * bsz, patch_size, f_patch_size)

            if i == 0:
                x_size = x_i_size
            x_seqlen = len(x_patches[0])
            cap_seqlen = len(cap_out[0])
            x_freqs_cis = self.rope_embedder(torch.cat(x_pos_ids[:1], dim=0)).unsqueeze(0)
            cap_freqs_cis = self.rope_embedder(torch.cat(cap_pos_ids[:1], dim=0)).unsqueeze(0)
            x_embedded = x_embedder(torch.stack(x_patches))
            x_embedded[torch.stack(x_inner_pad_mask)] = self.x_pad_token
            embedded_x_list.append(x_embedded)
            cap_embedded = self.cap_embedder(torch.stack(cap_out))
            cap_embedded[torch.stack(cap_inner_pad_mask)] = self.cap_pad_token
            cap_embedded_list.append(cap_embedded)
            per_sample_kwargs.append(
                {
                    "x_seqlen": x_seqlen,
                    "cap_seqlen": cap_seqlen,
                    "x_freqs": x_freqs_cis,
                    "cap_freqs": cap_freqs_cis,
                    "x_attn_mask": torch.ones((1, x_seqlen), dtype=torch.bool, device=device),
                    "cap_attn_mask": torch.ones((1, cap_seqlen), dtype=torch.bool, device=device),
                }
            )
            x = cap_feats = None

        adaln_input = t.type_as(embedded_x_list[0])[:1]

        for layer in self.noise_refiner:
            for i, x_i in enumerate(embedded_x_list):
                kwargs = dict(
                    attn_mask=per_sample_kwargs[i]["x_attn_mask"],
                    freqs_cis=per_sample_kwargs[i]["x_freqs"],
                    adaln_input=adaln_input,
                )
                embedded_x_list[i] = layer(x_i, **kwargs)
                x_i = None

        for layer in self.context_refiner:
            for i, cap_i in enumerate(cap_embedded_list):
                cap_embedded_list[i] = layer(
                    cap_i,
                    attn_mask=per_sample_kwargs[i]["cap_attn_mask"],
                    freqs_cis=per_sample_kwargs[i]["cap_freqs"],
                )
                cap_i = None

        unified_list = []
        unified_freqs_list = []
        unified_attn_masks = []
        for i, (embedded_x, cap_embedded) in enumerate(zip(embedded_x_list, cap_embedded_list)):
            unified_list.append(
                torch.cat([embedded_x, cap_embedded], dim=1)
            )
            embedded_x_list[i] = None
        for i in range(num_noise_samples):
            x_freqs_i = per_sample_kwargs[i]["x_freqs"]
            cap_freqs_i = per_sample_kwargs[i]["cap_freqs"]
            unified_freqs_i = torch.cat([x_freqs_i, cap_freqs_i], dim=1)
            unified_freqs_list.append(unified_freqs_i)
            unified_seqlen = per_sample_kwargs[i]["x_seqlen"] + per_sample_kwargs[i]["cap_seqlen"]
            unified_attn_masks.append(torch.ones((1, unified_seqlen), dtype=torch.bool, device=device))

        for layer in self.layers:
            if callback is not None:
                callback(-1, None, False, True)
            if pipeline is not None and getattr(pipeline, "_interrupt", False):
                return None

            for i, unified_i in enumerate(unified_list):
                unified_list[i] = layer(
                    unified_i,
                    attn_mask=unified_attn_masks[i],
                    freqs_cis=unified_freqs_list[i],
                    adaln_input=adaln_input,
                )
                unified_i = None

        output_list = []
        final_layer = self.all_final_layer[f"{patch_size}-{f_patch_size}"]
        for i in range(num_noise_samples):
            final_out = final_layer(unified_list[i], adaln_input)
            final_out = final_out[:, :per_sample_kwargs[i]["x_seqlen"]]
            unpatchified = self.unpatchify(final_out, x_size, patch_size, f_patch_size)
            output_list.append(unpatchified)
            final_out = None

        return output_list
