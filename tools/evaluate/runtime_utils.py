from __future__ import annotations

import io
import importlib
import logging
import math
import os
import sys
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout


# All transformers models in this toolkit use local_files_only=True — no network needed.
# Setting this permanently prevents the safetensors auto-conversion background thread
# from making network requests and printing spurious errors.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

QUIET_LOGGERS = [
    "huggingface_hub",
    "huggingface_hub.utils._http",
    "httpx",
    "httpcore",
    "transformers",
]


@contextmanager
def suppress_optional_model_noise():
    """Silence noisy third-party warnings/logging during optional model loading."""
    logger_states = {}
    for name in QUIET_LOGGERS:
        logger = logging.getLogger(name)
        logger_states[name] = logger.level
        logger.setLevel(logging.ERROR)

    out_buffer = io.StringIO()
    err_buffer = io.StringIO()

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Importing from timm\.models\.layers is deprecated.*",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"torch\.meshgrid: in an upcoming release.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Argument 'onesided' has been deprecated.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Using custom `forced_decoder_ids` from the \(generation\) config.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Transcription using a multilingual Whisper will default to language detection.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"A custom logits processor of type .* has been passed to `\.generate\(\)`.*",
                category=UserWarning,
            )
            with redirect_stdout(out_buffer), redirect_stderr(err_buffer):
                yield
    finally:
        for name, level in logger_states.items():
            logging.getLogger(name).setLevel(level)


def is_missing_metric(value) -> bool:
    try:
        return value is None or math.isnan(float(value))
    except (TypeError, ValueError):
        return value is None


def json_ready_metrics(metrics: dict) -> dict:
    """Convert NaN values to None so evaluation.json stays valid JSON."""
    cleaned = {}
    for key, value in metrics.items():
        cleaned[key] = None if is_missing_metric(value) else value
    return cleaned


def enable_offline_hf_mode():
    """Force local-cache-only behavior for Hugging Face loaders in evaluation."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


@contextmanager
def force_transformers_offline():
    """Temporarily set TRANSFORMERS_OFFLINE=1 to suppress safetensors auto-conversion network requests."""
    prev = os.environ.get("TRANSFORMERS_OFFLINE")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = prev


def load_local_whisper_pipeline():
    """Build a Whisper ASR pipeline from local cache only."""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    model_id = "openai/whisper-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_arg = 0 if device == "cuda" else -1

    with suppress_optional_model_noise(), force_transformers_offline():
        processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, local_files_only=True)
        if device == "cuda":
            model.to(device)
        return pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device_arg,
        )


def import_laion_clap_safely():
    """Import laion_clap without letting it parse this process's CLI arguments."""
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]]
        return importlib.import_module("laion_clap")
    finally:
        sys.argv = original_argv


@contextmanager
def force_torch_load_full_pickle():
    """
    Make third-party torch.load calls compatible with PyTorch 2.6+.

    laion_clap still expects the old torch.load default (weights_only=False).
    """
    import torch

    original_torch_load = torch.load

    def _compat_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = _compat_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def load_clap_checkpoint_compat(model, checkpoint_path: str):
    """
    Load a LAION-CLAP checkpoint with PyTorch 2.6+ compatibility.

    The upstream package does not handle torch.load(weights_only=True) and
    older checkpoints may include RoBERTa's non-parameter position_ids buffer.
    """
    from laion_clap.clap_module.factory import load_state_dict

    with force_torch_load_full_pickle():
        state_dict = load_state_dict(checkpoint_path, skip_params=True)

    state_dict.pop("text_branch.embeddings.position_ids", None)
    model.model.load_state_dict(state_dict, strict=False)
    return model
