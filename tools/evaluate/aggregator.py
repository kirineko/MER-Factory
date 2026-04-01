from __future__ import annotations

import math
from typing import Dict


DEFAULT_WEIGHTS = {
    "grounding": 0.45,  # clip/clap/asr
    "au": 0.10,         # au_f1
    "emotion": 0.0,    # placeholder for future emotion consistency metrics
    "consistency": 0.25, # nli entail - contra
    "temporal": 0.0,   # placeholder for future temporal metrics
    "style": 0.20,      # distinctness - repetition - toxicity
}


def _coerce_metric(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def aggregate_sample_metrics(m: Dict[str, float], weights: Dict[str, float] = DEFAULT_WEIGHTS) -> float:
    grounding_parts = []
    clip_v = _coerce_metric(m.get("clip_image_score"))
    if clip_v is not None:
        grounding_parts.append(clip_v)
    clap_v = _coerce_metric(m.get("clap_audio_score"))
    if clap_v is not None:
        grounding_parts.append(clap_v)
    asr_wer = _coerce_metric(m.get("asr_wer"))
    if asr_wer is not None:
        grounding_parts.append(1.0 - asr_wer)
    grounding = (
        max(0.0, min(1.0, sum(grounding_parts) / len(grounding_parts)))
        if grounding_parts else None
    )

    au = _coerce_metric(m.get("au_f1"))

    # emotion consistency placeholder (0.0..1.0)
    emotion = _coerce_metric(m.get("emotion_consistency"))

    # Use simplified consistency score (already computed in metrics)
    consistency = _coerce_metric(m.get("nli_consistency_score"))

    # temporal placeholder
    temporal = _coerce_metric(m.get("temporal_alignment"))

    # Style: favor distinct1 and distinct2, penalize repetition
    d1 = _coerce_metric(m.get("distinct1")) or 0.0
    d2 = _coerce_metric(m.get("distinct2")) or 0.0
    rep = _coerce_metric(m.get("repetition_rate")) or 0.0
    style = max(0.0, min(1.0, 0.5 * d1 + 0.5 * d2 - 0.5 * rep))

    weighted_sum = 0.0
    total_weight = 0.0
    for name, value in {
        "grounding": grounding,
        "au": au,
        "emotion": emotion,
        "consistency": consistency,
        "temporal": temporal,
        "style": style,
    }.items():
        if value is None:
            continue
        weight = weights[name]
        if weight <= 0:
            continue
        weighted_sum += weight * value
        total_weight += weight

    if total_weight == 0:
        return math.nan

    score = weighted_sum / total_weight
    return float(max(0.0, min(100.0, 100.0 * score)))

