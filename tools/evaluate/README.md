# MER-Factory Evaluation Suite

A comprehensive reference-free evaluation toolkit for MER-Factory outputs. This suite provides automated metrics to assess annotation quality without human ratings, supporting MER (full), audio, video, image, and AU analysis pipelines with graceful degradation when artifacts or dependencies are missing.

## Preflight Checklist

Before running evaluation, make sure the following prerequisites are ready. Most evaluation issues come from missing runtime dependencies rather than the command itself.

### Required Python packages

The evaluation pipeline depends on the following packages in addition to the core project dependencies:

- `torch`
- `torchaudio`
- `transformers`
- `open_clip_torch`
- `laion-clap`
- `opencc-python-reimplemented`

`torchaudio` is required by the CLAP audio grounding path. If it is missing or mismatched with your installed `torch`, `clap_audio_score` will silently fall back to a default value and become unreliable.

### Recommended install flow

From the project root:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

If `torch` and `torchaudio` version mismatch causes import/runtime errors, reinstall them as a matching pair:

```bash
uv pip install --python .venv/bin/python --reinstall torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0
```

### System and network prerequisites

- Internet access is needed the first time you load CLIP / CLAP / NLI / Whisper checkpoints
- If you are in a restricted network environment, use a Hugging Face mirror
- CPU is supported, but first-time model loading will be slow

### Optional: use Hugging Face mirror

If direct access to `huggingface.co` is unstable, set:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Then pre-download the models used by evaluation:

```bash
hf download openai/whisper-base
hf download microsoft/deberta-large-mnli
hf download roberta-base
hf download laion/CLIP-ViT-B-32-laion2B-s34B-b79K
hf download lukewys/laion_clap
```

If you do not have the CLI yet:

```bash
uv pip install --python .venv/bin/python "huggingface_hub[cli]"
```

### Verify the evaluation environment

Run these checks before the full evaluation:

```bash
python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__)"
python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='openai/whisper-base'); print('whisper ok')"
python tools/evaluate.py --help
```

If the second command fails, `asr_wer` will not be a real score. If `torchaudio` import fails, `clap_audio_score` will not be a real score.

## Overview

### Why automated, reference‑free evaluation?
- **Scalability**: Evaluate large datasets without human raters
- **Objectivity**: Measure grounding, consistency, and structure with reproducible metrics
- **Debuggability**: Identify failure modes (hallucination, poor grounding, weak AU alignment) quickly
- **Model‑agnostic**: Works across providers (Gemini, ChatGPT, Ollama, HuggingFace) and pipeline types
- **Real-time feedback**: Beautiful progress bars and color-coded results for immediate insights

## Model Dependencies

Evaluation is not a zero-download command. The first run may need to pull several Hugging Face checkpoints.

| Metric / Component | Hugging Face model | Why it is needed | Notes |
|--------------------|--------------------|------------------|-------|
| `clip_image_score` | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | Image-text grounding for peak frames | Loaded through `open_clip_torch` |
| `clap_audio_score` | `lukewys/laion_clap` | CLAP audio checkpoint (`630k-audioset-best.pt`) | Loaded with `hf_hub_download` |
| `clap_audio_score` | `roberta-base` | Text encoder used by `laion-clap` | CLAP also requires a working `torchaudio` runtime |
| `nli_consistency_score` | `microsoft/deberta-large-mnli` | Entailment / contradiction check across modalities | Used by the NLI scorer |
| `asr_wer` | `openai/whisper-base` | Whisper baseline for transcript verification | Requires processor / feature extractor files too |

If these checkpoints are not available, evaluation still runs, but some metrics degrade to placeholder values instead of real scores.

## Quick Start

Do not start with the command alone on a fresh machine. Complete the preflight section first, especially model downloads and `torchaudio` verification.

### Basic Usage
```bash
python tools/evaluate.py output/ --export-csv output/evaluation_summary.csv
```

### First-run recommendation

On a fresh machine, use this sequence:

```bash
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
python tools/evaluate.py output/ --export-csv output/evaluation_summary.csv
```

### With Verbose Output
```bash
python tools/evaluate.py output/ --export-csv output/evaluation_summary.csv --verbose
```

### Advanced Options
```bash
python tools/evaluate.py output/ \
    --export-csv output/evaluation_summary.csv \
    --write-per-sample \
    --verbose \
    --batch-size 16
```

**Performance**: Batch processing provides 3-8x speedup on GPU compared to single-sample mode. All evaluation functions (CLIP, CLAP, NLI, ASR) automatically detect and handle batch inputs.

## Supported Pipeline Types

The evaluation system automatically detects and adapts to different pipeline types:

| Pipeline Type | Detection Logic | Applicable Metrics |
|---------------|----------------|-------------------|
| **MER (full)** | `*_merr_data.json` present | All metrics (CLIP, CLAP, AU, NLI, ASR, Style) |
| **Audio** | `*_audio_analysis.json` present | CLAP, ASR WER, NLI, Style |
| **Video** | `*_video_analysis.json` present | CLIP, NLI, Style |
| **Image** | `*_image_analysis.json` present | CLIP, NLI, Style |
| **AU** | `*_au_analysis.json` present | AU alignment, Style |

## Metric Categories

### 🖼️ Grounding Metrics (Media ↔ Text)

#### CLIP Image-Text Score (`clip_image_score`)
- **Purpose**: Measures visual grounding between peak frame and text description
- **Method**: Cosine similarity via OpenAI CLIP (ViT-B-32)
- **Range**: 0-1 (normalized from -1 to 1)
- **Interpretation**: Higher scores indicate better visual alignment
- **Dependencies**: `open_clip_torch`, `torch`, `PIL`
- **Fallback**: Returns 0.0 if dependencies missing or no image available

#### CLAP Audio-Text Score (`clap_audio_score`)  
- **Purpose**: Measures audio-text alignment between WAV file and transcript/description
- **Method**: Cosine similarity via LAION-CLAP (HTSAT-base)
- **Range**: 0-1 (normalized from -1 to 1)
- **Interpretation**: Higher scores indicate better audio grounding
- **Dependencies**: `laion-clap`, `torch`, `torchaudio`
- **Fallback**: Returns 0.0 if dependencies missing or no audio available

#### ASR Word Error Rate (`asr_wer`)
- **Purpose**: Validates transcript quality against strong ASR baseline
- **Method**: WER comparison with Whisper/faster-whisper transcription
- **Range**: 0-1 (lower is better)
- **Interpretation**: Lower WER indicates more accurate transcription
- **Dependencies**: Hugging Face `transformers` pipeline with `openai/whisper-base`
- **Special Features**: 
  - Chinese language support with character-level tokenization
  - Traditional to Simplified Chinese normalization
  - Intelligent fallback between ASR engines

## What a "real" score requires

The evaluation script degrades gracefully, but that means some numbers may be placeholders if the environment is incomplete.

### `clap_audio_score`

For this metric to be meaningful, all of the following must be true:

- `*.wav` exists for the sample
- `laion-clap` loads successfully
- `torchaudio` works correctly
- the CLAP text/audio embedding path runs without exception

If these fail, the raw CLAP score falls back to `0.0`, and after normalization it may appear as `0.5` in the CSV. A CSV full of `0.5` CLAP values usually means the metric did not really run.

### `asr_wer`

For this metric to be meaningful, all of the following must be true:

- `*.wav` exists for the sample
- a transcript exists in the analysis output
- `openai/whisper-base` loads successfully

If Whisper fails to load, `asr_wer` falls back to `0.0`. In that case, `0.0` does **not** mean perfect transcription; it means the metric was unavailable.

### Quick interpretation tip

- `clap_audio_score == 0.5` for nearly every sample: likely fallback, not real CLAP alignment
- `asr_wer == 0.0` for nearly every sample: likely Whisper unavailable, not perfect ASR
- `Models initialized: ['clip', 'clap', 'nli']`: Whisper is missing
- `Models initialized: ['clip', 'clap', 'nli', 'whisper']`: all major metrics are available

### 😊 AU Alignment Metrics (Facial Expression ↔ Text)

#### AU Precision/Recall/F1 (`au_pr`, `au_re`, `au_f1`)
- **Purpose**: Matches textual AU descriptions to detected facial actions
- **Method**: Lexicon-based extraction + OpenFace AU intensity comparison
- **Range**: 0-1 (higher is better)
- **AU Lexicon**: Maps phrases like "smile" → AU12_r, "brow raiser" → AU01_r
- **Threshold**: Default 0.8 for AU presence detection
- **Data Sources**: 
  - OpenFace CSV files (`*_au_data.csv`)
  - Peak frame AU intensities from JSON
- **Fallback**: Returns 0.0 if no AU data or text available

### 🔗 Cross-Modal Consistency

#### NLI Consistency Score (`nli_consistency_score`, `nli_entail_rate`, `nli_contra_rate`)
- **Purpose**: Measures logical consistency between multimodal summary and unimodal descriptions
- **Method**: Natural Language Inference via DeBERTa-large-MNLI
- **Logic**: Entailment rate - Contradiction rate (only positive values kept)
- **Range**: 0-1 (higher is better)
- **Dependencies**: `transformers`, `torch`
- **Fallback**: Returns 0.0 if dependencies missing or insufficient text

### 📝 Text Quality & Style

#### Distinct N-grams (`distinct1`, `distinct2`)
- **Purpose**: Measures lexical diversity in generated text
- **Method**: Ratio of unique unigrams/bigrams to total
- **Range**: 0-1 (higher indicates more diversity)
- **Interpretation**: Low values may indicate repetitive generation

#### Repetition Rate (`repetition_rate`)
- **Purpose**: Detects overly repetitive text generation
- **Method**: 1 - (unique tokens / total tokens)
- **Range**: 0-1 (lower is better)
- **Interpretation**: Higher values indicate more repetition

#### Readability Score (`fkgl`)
- **Purpose**: Measures text complexity via Flesch-Kincaid Grade Level
- **Method**: Approximation based on sentence/word/syllable counts
- **Interpretation**: Extremely high/low values may indicate poor generation quality

### 🎯 Composite Score

The overall quality score (0-100) combines all metrics with carefully tuned weights:

```python
DEFAULT_WEIGHTS = {
    "grounding": 0.45,  # clip/clap/asr
    "au": 0.10,         # au_f1
    "emotion": 0.0,    # placeholder for future emotion consistency metrics
    "consistency": 0.25, # nli entail - contra
    "temporal": 0.0,   # placeholder for future temporal metrics
    "style": 0.20,      # distinctness - repetition - toxicity
}
```

**Score Interpretation**:
- 🎉 **80-100**: Excellent quality
- 👍 **60-79**: Good quality  
- 📈 **0-59**: Needs improvement

## File Structure & Data Loading

### Expected Directory Structure
```
output/
├── sample_00000000/
│   ├── sample_00000000_merr_data.json     # MER pipeline output
│   ├── sample_00000000_peak_frame.png     # Peak frame image
│   ├── sample_00000000.wav               # Audio file
│   ├── sample_00000000_au_data.csv       # OpenFace AU data
│   └── evaluation.json                   # Generated metrics
├── sample_00000001/
│   └── ...
└── evaluation_summary.csv                # Dataset-level results
```

### Artifact Discovery Logic

The `SampleArtifactPaths` class automatically discovers relevant files:

| File Pattern | Purpose | Priority |
|-------------|---------|----------|
| `*_merr_data.json` | MER pipeline output | Highest |
| `*_audio_analysis.json` | Audio-only pipeline | Medium |
| `*_video_analysis.json` | Video-only pipeline | Medium |
| `*_image_analysis.json` | Image-only pipeline | Medium |
| `*_au_analysis.json` | AU-only pipeline | Low |
| `*_peak_frame.{png,jpg}` | Peak frame image | - |
| `*.wav` | Audio file | - |
| `*_au_data.csv` | OpenFace AU data | - |

## Output Formats

### Per-Sample Output (`evaluation.json`)
```json
{
  "clip_image_score": 0.785,
  "clap_audio_score": 0.692,
  "asr_wer": 0.123,
  "au_pr": 0.834,
  "au_re": 0.756,
  "au_f1": 0.793,
  "nli_consistency_score": 0.667,
  "nli_entail_rate": 0.750,
  "nli_contra_rate": 0.083,
  "distinct1": 0.923,
  "distinct2": 0.887,
  "repetition_rate": 0.076,
  "fkgl": 8.2,
  "composite_score": 73.4
}
```

### Dataset Summary (`evaluation_summary.csv`)
- Sorted by `composite_score` (descending)
- All metrics included for statistical analysis
- Easy import into analysis tools

### Console Output
- 🏆 Top 10 performing samples table
- 📈 Overall statistics panel
- 📚 Metric explanations table
- Color-coded values (green/yellow/red thresholds)

## Advanced Configuration

### Custom Weights
Modify the aggregation weights in `aggregator.py`:
```python
custom_weights = {
    "grounding": 0.30,    # Emphasize grounding
    "au": 0.25,           # Increase AU importance  
    "consistency": 0.20,  # More consistency weight
    "style": 0.15,        # Higher style weight
    "emotion": 0.10,      # Reserve for future
    "temporal": 0.00,     # Disable temporal
}
```

### AU Presence Threshold
Adjust AU detection sensitivity:
```python
compute_au_alignment_metrics(
    au_csv_path=csv_path,
    peak_frame_index=frame_idx,
    peak_frame_au_text=text,
    presence_threshold=0.9  # Stricter threshold
)
```
