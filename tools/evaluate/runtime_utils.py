from __future__ import annotations

import io
import logging
import math
import os
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout


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


def load_local_whisper_pipeline():
    """Build a Whisper ASR pipeline from local cache only."""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    model_id = "openai/whisper-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_arg = 0 if device == "cuda" else -1

    with suppress_optional_model_noise():
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
