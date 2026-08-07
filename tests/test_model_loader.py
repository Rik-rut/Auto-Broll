from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch

from broll_agent.zimage.model_loader import (
    _FUSED_SPLIT_MAP,
    _HF_FILES,
    conv_state_dict,
    ensure_model_files,
)


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


def test_ensure_model_files_skips_existing(tmp_path):
    """If all files exist, no download should happen."""
    for _, local_name in _HF_FILES:
        (tmp_path / local_name).write_bytes(b"fake")

    with patch("huggingface_hub.hf_hub_download") as mock_download:
        ensure_model_files(tmp_path)
        mock_download.assert_not_called()


def test_ensure_model_files_downloads_missing(tmp_path):
    """Missing files should trigger download."""
    (tmp_path / _HF_FILES[0][1]).write_bytes(b"fake")

    def fake_download(repo_id, filename, local_dir, local_dir_use_symlinks):
        out = Path(local_dir) / Path(filename).name
        out.write_bytes(b"downloaded")
        return str(out)

    with patch("huggingface_hub.hf_hub_download", side_effect=fake_download):
        ensure_model_files(tmp_path)

    for _, local_name in _HF_FILES:
        assert (tmp_path / local_name).exists()


def test_hf_files_mapping():
    """Verify the HF file mapping has the expected structure."""
    assert len(_HF_FILES) == 4
    hf_paths = [hf for hf, _ in _HF_FILES]
    assert "ZImageTurbo_quanto_bf16_int8.safetensors" in hf_paths
    assert "Qwen3/qwen3_quanto_bf16_int8.safetensors" in hf_paths
