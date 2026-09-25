"""Prefill-only Qwen3-VL text encoder (language tower) for Qwen-Image 2.1.

Adapted from Wan2GP models/qwen21/text_encoder.py and qwen21_handler.py @
2345ae148f82740f66e82c41292dbbdd592e713d. The vision tower and ``lm_head`` are dropped because b-roll generation
is text-to-image only.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from .compat import check_abort
from .positions import Qwen3VLModel
from .text_layers import Qwen3VLTextDecoderLayer, Qwen3VLTextRotaryEmbedding, Qwen3VLTextRMSNorm


class TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config)

    def forward(self, input_ids, position_ids, visual_mask):
        hidden = self.embed_tokens(input_ids)
        positions = self.rotary_emb(hidden, position_ids)
        for layer in self.layers:
            check_abort()
            hidden = layer(hidden, position_embeddings=positions)
        return self.norm(hidden)


class Qwen3VLTextEncoder(nn.Module, Qwen3VLModel):
    """Language-only Qwen3-VL prefill encoder."""

    def __init__(self, config):
        super().__init__()
        text = dict(config["text_config"])
        if "rope_parameters" not in text:
            text["rope_parameters"] = {**text["rope_scaling"], "rope_theta": text["rope_theta"]}
        text_config = SimpleNamespace(**text)
        vision_config = SimpleNamespace(**dict(config["vision_config"]))
        self.config = SimpleNamespace(
            **{**config, "text_config": text_config, "vision_config": vision_config}
        )
        self.model = nn.Module()
        self.model.language_model = TextModel(text_config)

    def forward(self, input_ids, attention_mask, **kwargs):
        image_mask = input_ids == self.config.image_token_id
        types = image_mask.to(torch.int32)
        positions, _ = self.get_rope_index(input_ids, types, attention_mask=attention_mask)
        check_abort()
        hidden = self.model.language_model(input_ids, positions, image_mask)
        return SimpleNamespace(hidden_states=(hidden,))


_TEXT_PREFIXES = ("embed_tokens", "layers", "norm")


def _model_name(key: str) -> str:
    key = key.removeprefix("model.")
    if key.partition(".")[0] in _TEXT_PREFIXES:
        key = "language_model." + key
    return "model." + key


def _is_language_key(key: str) -> bool:
    return not key.removeprefix("model.").startswith("visual.")


def encoder_state_dict(state_dict, quantization_map=None, tied_weights_map=None):
    """Remap the encoder-only checkpoint (language_model.*/visual.*) to this module.

    Vision weights and the unused lm_head are dropped. mmgp calls this with
    (state_dict [, quantization_map [, tied_weights_map]]) depending on what the
    checkpoint carries, and accepts a 2- or 3-tuple back.
    """
    state = {
        _model_name(key): value
        for key, value in state_dict.items()
        if _is_language_key(key) and key != "lm_head.weight"
    }

    def remap(mapping):
        if mapping is None:
            return None
        out = {}
        for key, value in mapping.items():
            if key in ("lm_head", "lm_head.weight") or not _is_language_key(key):
                continue
            if isinstance(value, list):
                value = [_model_name(item) for item in value if item != "lm_head.weight"]
            out[_model_name(key)] = value
        return out

    return state, remap(quantization_map), remap(tied_weights_map)
