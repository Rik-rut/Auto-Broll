from __future__ import annotations

import json
from pathlib import Path

from accelerate import init_empty_weights

from broll_agent.qwen21.text_encoder import encoder_state_dict
from broll_agent.qwen21.transformer import QwenImage21Transformer2DModel

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "broll_agent"
    / "qwen21"
    / "configs"
    / "qwen_image_21_7B.json"
)


def test_encoder_state_dict_remaps_language_keys() -> None:
    state = {
        "language_model.embed_tokens.weight": 1,
        "language_model.layers.0.self_attn.q_proj.weight": 2,
        "visual.blocks.0.attn.qkv.weight": 3,
        "lm_head.weight": 4,
    }
    remapped, _, _ = encoder_state_dict(state)
    assert set(remapped) == {
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
    }


def test_encoder_state_dict_remaps_quant_map_and_drops_vision() -> None:
    state = {"language_model.layers.0.self_attn.q_proj.weight": 1}
    qmap = {
        "language_model.layers.0.self_attn.q_proj.weight": "qint8",
        "visual.blocks.0.attn.qkv.weight": "qint8",
    }
    tied = {"lm_head.weight": ["language_model.embed_tokens.weight"]}
    _, remapped_qmap, remapped_tied = encoder_state_dict(state, qmap, tied)
    assert remapped_qmap == {
        "model.language_model.layers.0.self_attn.q_proj.weight": "qint8"
    }
    assert remapped_tied == {}


def test_transformer_config_builds_expected_shape() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config = {key: value for key, value in config.items() if not key.startswith("_")}
    with init_empty_weights():
        model = QwenImage21Transformer2DModel(**config)
    assert len(model.transformer_blocks) == 32
    assert model.split_linear_modules_map["gate_up"]["mapped_modules"] == (
        "gate_layer",
        "proj",
    )


def test_text_layers_is_text_only() -> None:
    from broll_agent.qwen21 import text_layers

    assert hasattr(text_layers, "Qwen3VLTextDecoderLayer")
    assert not hasattr(text_layers, "Qwen3VLVisionModel")


def test_int8_convrot_handler_detects_and_converts() -> None:
    import torch

    from broll_agent.qwen21 import int8_convrot

    config = b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'
    state = {
        "layers.0.weight": torch.ones(4, 8, dtype=torch.int8),
        "layers.0.weight_scale": torch.ones(4, 1, dtype=torch.float32),
        "layers.0.comfy_quant": torch.tensor(list(config), dtype=torch.uint8),
    }
    info = int8_convrot.detect(state)
    assert info["matched"] is True
    assert info["kind"] == "int8_convrot"

    converted = int8_convrot.convert_to_quanto(state, default_dtype=torch.bfloat16)
    converted_state = converted["state_dict"]
    assert "layers.0.weight._data" in converted_state
    assert "layers.0.weight._scale" in converted_state
    assert converted["quant_map"]["layers.0"]["weights"] == "qint8_convrot"
