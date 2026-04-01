"""Evaluation toolkit for MER-Factory outputs.

Modules are organized as follows:
- loaders: discover and read MER outputs and related artifacts
- metrics_au: AU grounding and alignment metrics
- metrics_text: simple textual style metrics
- metrics_grounding: optional CLIP/CLAP/NLI/ASR metrics with lazy imports
- aggregator: compose per-sample metrics into a composite score
"""

from .loaders import (
    SampleArtifactPaths,
    find_samples,
    load_mer_output,
)
from .metrics_au import compute_au_alignment_metrics
from .metrics_text import compute_text_style_metrics
from .metrics_grounding import (
    compute_clip_image_text_score,
    compute_clap_audio_text_score,
    compute_nli_consistency_scores,
    compute_asr_wer,
)
from .aggregator import aggregate_sample_metrics
from .runtime_utils import (
    suppress_optional_model_noise,
    load_local_whisper_pipeline,
    force_transformers_offline,
    import_laion_clap_safely,
    load_clap_checkpoint_compat,
)
import torch

# Add model initialization helper
def initialize_models():
    """Initialize all models once for efficient batch evaluation."""
    models = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize CLIP models
    print("Loading CLIP model...")
    try:
        with suppress_optional_model_noise():
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()
        model.to(device)
        models['clip'] = {
            'model': model,
            'preprocess': preprocess,
            'tokenizer': tokenizer
        }
        print(f"✓ CLIP model loaded on {device}")
    except Exception as e:
        models['clip'] = None
        print(f"✗ CLIP model failed to load: {e}")
    
    # Initialize CLAP model
    print("Loading CLAP model...")
    try:
        from huggingface_hub import hf_hub_download
        clap_ckpt = hf_hub_download(repo_id="lukewys/laion_clap", filename="630k-audioset-best.pt")
        with suppress_optional_model_noise(), force_transformers_offline():
            laion_clap = import_laion_clap_safely()
            model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
            load_clap_checkpoint_compat(model, clap_ckpt)
        model.eval()
        model.to(device)
        models['clap'] = model
        print(f"✓ CLAP model loaded on {device}")
    except Exception as e:
        models['clap'] = None
        print(f"✗ CLAP model failed to load: {e}")
    
    # Initialize NLI model
    print("Loading NLI model...")
    try:
        with suppress_optional_model_noise(), force_transformers_offline():
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            model_name = "microsoft/deberta-large-mnli"
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
        model.eval()
        model.to(device)
        models['nli'] = {
            'model': model,
            'tokenizer': tokenizer
        }
        print(f"✓ NLI model loaded on {device}")
    except Exception as e:
        models['nli'] = None
        print(f"✗ NLI model failed to load: {e}")
    
    # Initialize Whisper model using HuggingFace pipeline
    print("Loading Whisper model...")
    try:
        pipe = load_local_whisper_pipeline()
        models['whisper'] = pipe
        print(f"✓ Whisper model loaded on {device}")
    except Exception as e:
        models['whisper'] = None
        print(f"✗ Whisper model failed to load: {e}")
    
    return models

__all__ = [
    "SampleArtifactPaths",
    "find_samples",
    "load_mer_output",
    "compute_au_alignment_metrics",
    "compute_text_style_metrics",
    "compute_clip_image_text_score",
    "compute_clap_audio_text_score",
    "compute_nli_consistency_scores",
    "compute_asr_wer",
    "aggregate_sample_metrics",
    "initialize_models",
]
