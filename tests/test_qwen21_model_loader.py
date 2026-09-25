from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from broll_agent.config import Settings
from broll_agent.qwen21 import model_loader


def test_hf_files_manifest() -> None:
    by_path = {entry.hf_path: entry for entry in model_loader._HF_FILES}
    assert model_loader._TRANSFORMER_WEIGHTS in by_path
    assert model_loader._VAE_WEIGHTS in by_path
    assert model_loader._SCHEDULER_CONFIG in by_path
    assert model_loader._LORA_WEIGHTS in by_path
    assert model_loader._ENCODER_WEIGHTS in by_path
    assert by_path[model_loader._TRANSFORMER_WEIGHTS].repo_id == model_loader._REPO_ID
    assert by_path[model_loader._ENCODER_WEIGHTS].repo_id == model_loader._ENCODER_REPO_ID
    assert by_path[model_loader._ENCODER_WEIGHTS].revision == model_loader._ENCODER_REVISION
    assert all(entry.local_path == entry.hf_path for entry in model_loader._HF_FILES)


def test_ensure_model_files_skips_existing(tmp_path: Path) -> None:
    for entry in model_loader._HF_FILES:
        target = tmp_path / entry.local_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fake")

    with patch("huggingface_hub.hf_hub_download") as mock_download:
        model_loader.ensure_model_files(tmp_path)
        mock_download.assert_not_called()


def test_ensure_model_files_downloads_missing(tmp_path: Path) -> None:
    for entry in model_loader._HF_FILES[:-1]:
        target = tmp_path / entry.local_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fake")

    def fake_download(repo_id, filename, revision, local_dir):
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"downloaded")
        return str(out)

    with patch("huggingface_hub.hf_hub_download", side_effect=fake_download) as mock_download:
        model_loader.ensure_model_files(tmp_path)

    assert mock_download.call_count == 1
    for entry in model_loader._HF_FILES:
        assert (tmp_path / entry.local_path).exists()


class _FakeTransformer:
    split_linear_modules_map = {"gate_up": {}}

    def preprocess_loras(self, model_type, state_dict):
        return state_dict


class _FakeTextEncoder:
    def __init__(self, config):
        self.config = config

    def eval(self):
        return self

    def requires_grad_(self, flag):
        return self


def test_load_qwen21_pipeline_wires_components(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "qwen model"
    cfg = Settings(
        _env_file=None,
        qwen_model_dir=model_dir,
        qwen_quanto_int8_kernel=False,
    )
    monkeypatch.setattr(model_loader, "ensure_model_files", lambda path: None)
    monkeypatch.setattr(model_loader, "Qwen3VLTextEncoder", _FakeTextEncoder)
    monkeypatch.setattr(model_loader, "load_processor", lambda *a, **k: object())
    model_loader.offload.shared_state.pop("_attention", None)

    fake_transformer = _FakeTransformer()
    calls: dict = {}

    def fake_fast_load(
        path,
        writable_tensors=True,
        modelClass=None,
        defaultConfigPath=None,
        default_dtype=None,
    ):
        calls.setdefault("fast_load", []).append((Path(path), modelClass))
        if modelClass is model_loader.QwenImage21Transformer2DModel:
            return fake_transformer
        return object()

    monkeypatch.setattr(model_loader.offload, "fast_load_transformers_model", fake_fast_load)
    monkeypatch.setattr(
        model_loader.offload,
        "load_model_data",
        lambda *a, **k: calls.setdefault("load_model_data", []).append((a, k)),
    )
    monkeypatch.setattr(
        model_loader.offload,
        "profile",
        lambda pipe, **k: calls.setdefault("profile", (pipe, k)),
    )
    monkeypatch.setattr(
        model_loader.offload,
        "load_loras_into_model",
        lambda *a, **k: calls.setdefault("lora", (a, k)),
    )

    class FakePipeline:
        def __init__(self, transformer, text_encoder, vae, processor, scheduler_config):
            self.transformer = transformer
            self.text_encoder = text_encoder
            self.vae = vae
            self.processor = processor
            self.scheduler_config = scheduler_config

    monkeypatch.setattr(model_loader, "Qwen21Pipeline", FakePipeline)

    encoder_dir = model_dir / model_loader._ENCODER_FOLDER
    encoder_dir.mkdir(parents=True)
    (encoder_dir / "config_legacy.json").write_text("{}", encoding="utf-8")

    pipeline = model_loader.load_qwen21_pipeline(cfg)

    fast_loads = calls["fast_load"]
    assert [entry[0].name for entry in fast_loads] == [
        model_loader._TRANSFORMER_WEIGHTS,
        "qwen_image_21_vae.safetensors",
    ]
    pipe_dict, profile_kwargs = calls["profile"]
    assert set(pipe_dict) == {"transformer", "text_encoder", "vae"}
    assert profile_kwargs["loras"] == ["transformer"]
    lora_args, lora_kwargs = calls["lora"]
    assert Path(lora_args[1]).name == Path(model_loader._LORA_WEIGHTS).name
    assert lora_kwargs["lora_multi"] == [1.0]
    assert lora_kwargs["activate_all_loras"] is True
    assert pipeline.scheduler_config == model_dir / model_loader._SCHEDULER_CONFIG
    assert pipeline.transformer is fake_transformer
    assert pipeline.vae is not fake_transformer
