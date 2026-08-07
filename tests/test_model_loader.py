from __future__ import annotations

import torch

from broll_agent.zimage.model_loader import conv_state_dict, _FUSED_SPLIT_MAP


def test_conv_state_dict_maps_final_layer():
    sd = {
        "x_embedder.weight": torch.zeros(1),
        "final_layer.linear.weight": torch.zeros(1),
    }
    result = conv_state_dict(sd)
    assert "all_final_layer.2-1.linear.weight" in result


def test_conv_state_dict_maps_x_embedder():
    sd = {"x_embedder.weight": torch.zeros(1)}
    result = conv_state_dict(sd)
    assert "all_x_embedder.2-1.weight" in result


def test_conv_state_dict_maps_attention_keys():
    sd = {
        "x_embedder.weight": torch.zeros(1),
        "layers.0.attention.out.bias": torch.zeros(1),
        "layers.0.attention.k_norm.weight": torch.zeros(1),
        "layers.0.attention.q_norm.weight": torch.zeros(1),
        "layers.0.attention.out.weight": torch.zeros(1),
    }
    result = conv_state_dict(sd)
    assert "layers.0.attention.to_out.0.bias" in result
    assert "layers.0.attention.norm_k.weight" in result
    assert "layers.0.attention.norm_q.weight" in result
    assert "layers.0.attention.to_out.0.weight" in result


def test_conv_state_dict_strips_model_prefix():
    sd = {"model.diffusion_model.x_embedder.weight": torch.zeros(1)}
    result = conv_state_dict(sd)
    assert "all_x_embedder.2-1.weight" in result


def test_conv_state_dict_passthrough_when_no_matching_keys():
    sd = {"some.other.key": torch.zeros(1)}
    result = conv_state_dict(sd)
    assert result == sd


def test_fused_split_map_has_qkv():
    assert "attention.to_qkv" in _FUSED_SPLIT_MAP
    assert _FUSED_SPLIT_MAP["attention.to_qkv"]["mapped_modules"] == (
        "attention.to_q",
        "attention.to_k",
        "attention.to_v",
    )
