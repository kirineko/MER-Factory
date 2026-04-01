from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import typer
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate import (
    find_samples,
    load_mer_output,
    compute_text_style_metrics,
    compute_au_alignment_metrics,
    compute_clip_image_text_score,
    compute_clap_audio_text_score,
    compute_nli_consistency_scores,
    compute_asr_wer,
    aggregate_sample_metrics,
    initialize_models,
)
from tools.evaluate.runtime_utils import is_missing_metric, json_ready_metrics


app = typer.Typer(add_completion=False, help="Evaluate MER-Factory outputs (reference-free metrics)")
console = Console()

def _detect_sample_type(sample) -> str:
    if getattr(sample, "mer_json", None):
        return "MER"
    if getattr(sample, "audio_json", None):
        return "audio"
    if getattr(sample, "video_json", None):
        return "video"
    if getattr(sample, "image_json", None):
        return "image"
    if getattr(sample, "au_json", None):
        return "au"
    return "unknown"


def _prepare_sample_data(sample):
    """Extract and prepare data from a sample for evaluation."""
    # Load available analysis JSON (prefer MER, else audio/video/image/AU)
    mer = {}
    candidate = sample.mer_json or sample.audio_json or sample.video_json or sample.image_json or sample.au_json
    if candidate and candidate.exists():
        try:
            mer = load_mer_output(candidate)
        except Exception:
            mer = {}

    # Extract fields used for metrics
    final_summary = mer.get("final_summary", "")
    coarse_desc = mer.get("coarse_descriptions_at_peak", {}) or {}
    
    # Safely extract descriptions
    peak_frame_visual_description = coarse_desc.get("visual_objective", "") or mer.get("image_visual_description", "")
    video_description = coarse_desc.get("video_content", "") or mer.get("llm_video_summary", "")
    audio_desc = coarse_desc.get("audio_analysis", "")
    
    # Extract transcript from audio_analysis (first line before \n)
    transcript = audio_desc.split('\n')[0] if audio_desc else ""
    peak_info = mer.get("overall_peak_frame_info") or {}
    peak_frame_index = peak_info.get("frame_number")
    peak_frame_au_text = ""
    peak_frame_au_text_source = None
    for source, value in [
        ("llm_au_description", mer.get("llm_au_description", "")),
        ("peak_frame_au_description", mer.get("peak_frame_au_description", "")),
        ("au_text_description", mer.get("au_text_description", "")),
        ("coarse_visual_expression", coarse_desc.get("visual_expression", "")),
    ]:
        if value:
            peak_frame_au_text = value
            peak_frame_au_text_source = source
            break

    return {
        'mer': mer,
        'final_summary': final_summary,
        'peak_frame_visual_description': peak_frame_visual_description,
        'video_description': video_description,
        'audio_desc': audio_desc,
        'transcript': transcript,
        'peak_info': peak_info,
        'peak_frame_index': peak_frame_index,
        'peak_frame_au_text': peak_frame_au_text,
        'peak_frame_au_text_source': peak_frame_au_text_source,
    }


@app.command()
def run(
    output_root: str = typer.Argument(..., help="Path to output directory containing per-sample folders"),
    export_csv: Optional[str] = typer.Option(None, help="Path to write evaluation_summary.csv"),
    write_per_sample: bool = typer.Option(True, help="Write evaluation.json in each sample folder"),
    verbose: bool = typer.Option(False, help="Print reasons when metrics are 0 due to missing deps/artifacts/text"),
    filter_type: Optional[str] = typer.Option("mer", "--type", help="Filter by sample type: mer, audio, video, image, au, or 'all' (default: mer)"),
    batch_size: int = typer.Option(8, help="Batch size for model inference (default: 8, use 1 to disable batching)"),
):
    root = Path(output_root)
    rows: List[Dict] = []
    
    # Initialize all models once at the beginning
    console.print("🚀 [bold blue]Initializing models[/bold blue] (this may take a moment)...", style="cyan")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Loading models...", total=None)
        models = initialize_models()
        progress.update(task, completed=True)
    
    initialized_models = [k for k, v in models.items() if v is not None]
    console.print(f"✅ [bold green]Models initialized:[/bold green] {initialized_models}")

    # Get all samples first
    all_samples = list(find_samples(root))
    
    # Normalize and validate filter_type
    if filter_type:
        filter_type = filter_type.lower()
        valid_types = ["mer", "audio", "video", "image", "au", "all"]
        if filter_type not in valid_types:
            console.print(f"❌ [bold red]Invalid type:[/bold red] '{filter_type}'. Must be one of: {', '.join(valid_types)}", style="red")
            raise typer.Exit(code=1)
    
    # Filter samples by type if specified
    if filter_type and filter_type != "all":
        samples = [s for s in all_samples if _detect_sample_type(s).lower() == filter_type]
        console.print(f"📊 [bold yellow]Found {len(samples)} {filter_type.upper()} samples to evaluate[/bold yellow] (filtered from {len(all_samples)} total)")
    else:
        samples = all_samples
        console.print(f"📊 [bold yellow]Found {len(samples)} samples to evaluate[/bold yellow]")
    
    # Exit if no samples found
    if not samples:
        console.print("⚠️ [bold yellow]No samples found matching the criteria[/bold yellow]")
        return
    
    # Show batch size info
    if batch_size > 1:
        console.print(f"⚡ [bold cyan]Batch inference enabled[/bold cyan] with batch size: {batch_size}")
    else:
        console.print(f"🔄 [yellow]Processing samples individually[/yellow]")
    
    # Process samples with beautiful progress bar
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("🔍 Processing samples", total=len(samples))
        
        # Process samples in batches
        for batch_start in range(0, len(samples), batch_size):
            batch_end = min(batch_start + batch_size, len(samples))
            batch_samples = samples[batch_start:batch_end]
            
            # Prepare data for all samples in batch
            batch_data = []
            for sample in batch_samples:
                data = _prepare_sample_data(sample)
                data['sample'] = sample
                data['sample_type'] = _detect_sample_type(sample)
                batch_data.append(data)
            
            # Collect batch inputs for model inference
            clip_image_paths = []
            clip_texts = []
            clap_audio_paths = []
            clap_texts = []
            nli_premises = []
            nli_hypotheses_list = []
            asr_transcripts = []
            asr_audio_paths = []
            
            for data in batch_data:
                # CLIP inputs
                img_path = None
                if getattr(data['sample'], "peak_frame_image", None) and data['sample'].peak_frame_image.exists():
                    img_path = str(data['sample'].peak_frame_image)
                clip_image_paths.append(img_path)
                clip_texts.append(
                    (data['peak_frame_visual_description'] or data['video_description'] or data['final_summary']) or None
                )
                
                # CLAP inputs
                audio_path = str(data['sample'].audio_wav) if data['sample'].audio_wav else None
                text = (data['transcript'] + " " + data['audio_desc']).strip() if (data['transcript'] or data['audio_desc']) else None
                clap_audio_paths.append(audio_path)
                clap_texts.append(text)
                
                # NLI inputs
                premise = data['final_summary'] or data['video_description'] or data['audio_desc'] or data['peak_frame_au_text']
                hypotheses = [t for t in [data['video_description'], data['audio_desc'], data['peak_frame_au_text']] if t]
                nli_premises.append(premise)
                nli_hypotheses_list.append(hypotheses)
                
                # ASR WER inputs
                asr_transcripts.append(data['transcript'])
                asr_audio_paths.append(str(data['sample'].audio_wav) if data['sample'].audio_wav else None)
            
            # Call unified functions (they handle batching internally)
            clip_models = models.get('clip')
            if clip_models:
                clip_scores = compute_clip_image_text_score(
                    clip_image_paths, clip_texts,
                    clip_model=clip_models['model'],
                    clip_preprocess=clip_models['preprocess'],
                    clip_tokenizer=clip_models['tokenizer']
                )
            else:
                clip_scores = [0.0] * len(batch_samples)
            
            clap_model = models.get('clap')
            clap_scores = compute_clap_audio_text_score(
                clap_audio_paths, clap_texts, clap_model=clap_model
            )
            
            nli_models = models.get('nli')
            if nli_models:
                nli_results = compute_nli_consistency_scores(
                    nli_premises, nli_hypotheses_list,
                    nli_model=nli_models['model'],
                    nli_tokenizer=nli_models['tokenizer']
                )
            else:
                nli_results = [{"nli_consistency_score": 0.0, "nli_entail_rate": 0.0, "nli_contra_rate": 0.0}] * len(batch_samples)
            
            whisper_model = models.get('whisper')
            asr_wer_scores = compute_asr_wer(
                asr_transcripts, asr_audio_paths, whisper_model=whisper_model
            )
            
            # Process results for each sample in batch
            for idx, (data, clip_score, clap_score, nli_result, asr_wer) in enumerate(
                zip(batch_data, clip_scores, clap_scores, nli_results, asr_wer_scores)
            ):
                sample = data['sample']
                sample_type = data['sample_type']
                final_summary = data['final_summary']
                peak_frame_au_text = data['peak_frame_au_text']
                peak_frame_au_text_source = data['peak_frame_au_text_source']
                peak_info = data['peak_info']
                peak_frame_index = data['peak_frame_index']
                transcript = data['transcript']
                audio_desc = data['audio_desc']
                video_description = data['video_description']
                
                progress.update(task, description=f"🔍 Processing {sample.sample_id}")
                
                metrics: Dict[str, float] = {}
                reasons: Dict[str, str] = {}

                # Pre-checks for artifacts/text (for verbose reporting)
                img_path = None
                if getattr(sample, "peak_frame_image", None) and sample.peak_frame_image.exists():
                    img_path = str(sample.peak_frame_image)
                if not img_path:
                    reasons["clip_image_score"] = "missing peak frame image"
                elif not clip_models:
                    reasons["clip_image_score"] = "CLIP model unavailable"

                if not getattr(sample, "audio_wav", None):
                    reasons["clap_audio_score"] = "missing wav"
                elif not (transcript or audio_desc):
                    reasons["clap_audio_score"] = "missing transcript/audio_description"
                elif not clap_model:
                    reasons["clap_audio_score"] = "CLAP model unavailable"

                if not getattr(sample, "audio_wav", None):
                    reasons["asr_wer"] = "missing wav"
                elif not transcript:
                    reasons["asr_wer"] = "missing transcript in audio_analysis"
                elif not whisper_model:
                    reasons["asr_wer"] = "Whisper model unavailable"

                has_peak_intensities = bool(peak_info.get("top_aus_intensities"))
                has_reliable_au_text = peak_frame_au_text_source == "llm_au_description"
                if not getattr(sample, "au_csv", None) and not has_peak_intensities:
                    reasons["au_f1"] = "missing AU CSV or top_aus_intensities"
                elif peak_frame_index is None and not has_peak_intensities:
                    reasons["au_f1"] = "missing peak frame index"
                elif not peak_frame_au_text:
                    reasons["au_f1"] = "missing AU text description"
                elif not has_reliable_au_text:
                    reasons["au_f1"] = (
                        f"AU text source '{peak_frame_au_text_source}' is OpenFace-derived and would make the metric circular; "
                        "rerun MER with the current code to save llm_au_description"
                    )

                nli_premise = final_summary or video_description or audio_desc or peak_frame_au_text
                nli_hypotheses = [t for t in [video_description, audio_desc, peak_frame_au_text] if t]
                if not (nli_premise and nli_hypotheses):
                    reasons["nli"] = "insufficient text for NLI"
                elif not nli_models:
                    reasons["nli"] = "NLI model unavailable"

                # Report missing data as warnings if verbose mode is enabled
                if verbose:
                    missing: List[str] = []
                    if sample_type == "MER":
                        for key in ["clip_image_score", "clap_audio_score", "asr_wer", "au_f1", "nli"]:
                            if key in reasons:
                                missing.append(f"{key}: {reasons[key]}")
                    elif sample_type == "audio":
                        for key in ["clap_audio_score", "asr_wer", "nli"]:
                            if key in reasons:
                                missing.append(f"{key}: {reasons[key]}")
                    elif sample_type == "video":
                        for key in ["clip_image_score", "nli"]:
                            if key in reasons:
                                missing.append(f"{key}: {reasons[key]}")
                    elif sample_type == "image":
                        for key in ["clip_image_score", "nli"]:
                            if key in reasons:
                                missing.append(f"{key}: {reasons[key]}")
                    elif sample_type == "au":
                        for key in ["au_f1"]:
                            if key in reasons:
                                missing.append(f"{key}: {reasons[key]}")
                    if missing:
                        console.print(
                            f"⚠️  [yellow][{sample.sample_id}] metric caveats:[/yellow] "
                            + "; ".join(missing)
                        )

                # Text style metrics (cheap, always on)
                style_m = compute_text_style_metrics(final_summary)
                metrics.update(style_m)

                # AU alignment metrics (cheap)
                if "au_f1" in reasons:
                    au_m = {"au_pr": math.nan, "au_re": math.nan, "au_f1": math.nan}
                else:
                    au_m = compute_au_alignment_metrics(
                        str(sample.au_csv) if sample.au_csv else None,
                        peak_frame_index,
                        peak_frame_au_text,
                        peak_au_intensities=peak_info.get("top_aus_intensities"),
                    )
                metrics.update(au_m)

                # Use batched inference results
                metrics["clip_image_score"] = clip_score
                metrics["clap_audio_score"] = clap_score
                metrics.update(nli_result)
                metrics["asr_wer"] = asr_wer

                # Normalize CLIP and CLAP scores for consistent 0-1 range in saved metrics
                if "clip_image_score" in metrics and not is_missing_metric(metrics["clip_image_score"]):
                    raw_clip = metrics["clip_image_score"]
                    metrics["clip_image_score"] = max(0.0, (raw_clip + 1.0) / 2.0)
                
                if "clap_audio_score" in metrics and not is_missing_metric(metrics["clap_audio_score"]):
                    raw_clap = metrics["clap_audio_score"]
                    metrics["clap_audio_score"] = max(0.0, (raw_clap + 1.0) / 2.0)

                # Composite score
                metrics["composite_score"] = aggregate_sample_metrics(metrics)

                # Persist per-sample
                if write_per_sample:
                    try:
                        with (sample.sample_dir / "evaluation.json").open("w", encoding="utf-8") as f:
                            json.dump(json_ready_metrics(metrics), f, indent=2)
                    except Exception:
                        pass

                row = {"sample_id": sample.sample_id, **metrics}
                rows.append(row)
                
                # Update progress
                progress.advance(task)

    df = pd.DataFrame(rows).sort_values("composite_score", ascending=False)
    if export_csv:
        out = Path(export_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        console.print(f"💾 [green]Results exported to:[/green] {out}")

    # Print a beautiful summary with key metrics only
    key_columns = ['sample_id', 'clip_image_score', 'clap_audio_score', 'au_f1', 
                   'nli_consistency_score', 'asr_wer', 'composite_score']
    available_columns = [col for col in key_columns if col in df.columns]
    
    # Create a rich table for the summary
    table_title = "🏆 Top Performing Samples"
    if filter_type and filter_type != "all":
        table_title = f"🏆 Top Performing {filter_type.upper()} Samples"
    table = Table(title=table_title, show_header=True, header_style="bold magenta")
    
    # Add columns with custom styling
    for col in available_columns:
        if col == 'sample_id':
            table.add_column("Sample ID", style="cyan", no_wrap=True)
        elif col == 'clip_image_score':
            table.add_column("🖼️ CLIP", justify="right", style="blue")
        elif col == 'clap_audio_score':
            table.add_column("🔊 CLAP", justify="right", style="green")
        elif col == 'au_f1':
            table.add_column("😊 AU F1", justify="right", style="yellow")
        elif col == 'nli_consistency_score':
            table.add_column("🔗 NLI", justify="right", style="purple")
        elif col == 'asr_wer':
            table.add_column("🎙️ WER", justify="right", style="red")
        elif col == 'composite_score':
            table.add_column("🎯 Overall", justify="right", style="bold white")
    
    # Add top 10 rows to the table
    top_samples = df[available_columns].head(min(10, len(df)))
    for _, row in top_samples.iterrows():
        row_data = []
        for col in available_columns:
            if col == 'sample_id':
                row_data.append(str(row[col]))
            elif col == 'asr_wer':
                # For WER, lower is better, so color accordingly
                value = row[col]
                if is_missing_metric(value):
                    row_data.append("[dim]N/A[/dim]")
                    continue
                if value <= 0.1:
                    row_data.append(f"[green]{value:.3f}[/green]")
                elif value <= 0.3:
                    row_data.append(f"[yellow]{value:.3f}[/yellow]")
                else:
                    row_data.append(f"[red]{value:.3f}[/red]")
            elif col in ['clip_image_score', 'clap_audio_score']:
                value = row[col]
                if is_missing_metric(value):
                    row_data.append("[dim]N/A[/dim]")
                    continue
                if value >= 0.8:
                    row_data.append(f"[green]{value:.3f}[/green]")
                elif value >= 0.6:
                    row_data.append(f"[yellow]{value:.3f}[/yellow]")
                else:
                    row_data.append(f"[red]{value:.3f}[/red]")
            else:
                # For other metrics, higher is better
                value = row[col]
                if is_missing_metric(value):
                    row_data.append("[dim]N/A[/dim]")
                    continue
                if value >= 0.8:
                    row_data.append(f"[green]{value:.3f}[/green]")
                elif value >= 0.6:
                    row_data.append(f"[yellow]{value:.3f}[/yellow]")
                else:
                    row_data.append(f"[red]{value:.3f}[/red]")
        table.add_row(*row_data)
    
    console.print(table)
    
    # Overall statistics panel
    overall_score = df['composite_score'].mean()
    if overall_score >= 80:
        score_color = "green"
        score_emoji = "🎉"
    elif overall_score >= 60:
        score_color = "yellow" 
        score_emoji = "👍"
    else:
        score_color = "red"
        score_emoji = "📈"
        
    stats_text = f"{score_emoji} Overall Composite Score: [{score_color}]{overall_score:.2f}[/{score_color}]\n"
    stats_text += f"📊 Total Samples: {len(df)}\n"
    stats_text += f"🥇 Best Sample: {df.iloc[0]['sample_id']} ({df.iloc[0]['composite_score']:.2f})"
    
    console.print(Panel(stats_text, title="📈 Summary Statistics", border_style="blue"))
    
    # Show metric explanations
    explanations = Table(title="📚 Metric Explanations", show_header=True, header_style="bold cyan")
    explanations.add_column("Metric", style="bold")
    explanations.add_column("Description", style="italic")
    explanations.add_column("Range", justify="center")
    explanations.add_column("Better", justify="center")
    
    explanations.add_row("🖼️ CLIP Image Score", "How well image matches text", "0-1", "Higher")
    explanations.add_row("🔊 CLAP Audio Score", "How well audio matches text", "0-1", "Higher")
    explanations.add_row("😊 AU F1", "Facial expression accuracy", "0-1", "Higher")
    explanations.add_row("🔗 NLI Consistency", "Text logical consistency", "0-1", "Higher")
    explanations.add_row("🎙️ ASR WER", "Speech recognition error rate", "0-1", "Lower")
    explanations.add_row("🎯 Composite Score", "Overall quality", "0-100", "Higher")
    
    console.print(explanations)


if __name__ == "__main__":
    app()
