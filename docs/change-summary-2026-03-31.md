# Change Summary - 2026-03-31

## Scope

This change set focuses on three areas:

1. Gemini model selection can now be overridden from the CLI.
2. Hugging Face multimodal integrations were stabilized for local-path loading and GPU memory cleanup.
3. OpenFace Windows binaries can now be called reliably from WSL.

## Changes

### Gemini CLI model override

- Added `--gemini-model` / `-gm` to the CLI in `main.py`.
- Added `gemini_model` to `AppConfig` in `utils/config.py`.
- Wired the selected Gemini model through `mer_factory/models/__init__.py`.
- Updated `mer_factory/models/api_models/gemini.py` to use the requested model name instead of a hardcoded default.

### Hugging Face model stability

- Fixed Hugging Face model registry lookup in `mer_factory/models/hf_models/__init__.py` so local paths and normalized model IDs resolve correctly.
- Fixed `Qwen2.5-Omni` generation in `mer_factory/models/hf_models/qwen2_5_omni.py` by preserving `input_ids` before tensor dictionaries are moved to device.
- Added explicit `pad_token_id` and per-request CUDA cache cleanup to `mer_factory/models/hf_models/qwen2_5_omni.py`.
- Added per-request CUDA cache cleanup to `mer_factory/models/hf_models/gemma_multimodal.py`.

### OpenFace WSL integration

- Updated `tools/openface_adapter.py` to:
  - convert WSL paths to Windows paths when launching a Windows `FeatureExtraction.exe`,
  - run OpenFace from its installation directory so relative model assets resolve correctly.
- Updated `test/test_openface.py` with the same path conversion logic and working-directory handling.
- Added CSV existence logging to make OpenFace test verification clearer.

## Verification

- Verified Python syntax with `py_compile` on the updated Hugging Face model files.
- Verified OpenFace test flow creates the expected CSV output under WSL when using the Windows binary.

## Commit Notes

- `dataset/` and `test_input/` are intentionally excluded from the commit.
