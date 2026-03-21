"""
Training Data Generator for Fine-Tuning

Creates training datasets from your existing scene analyses for:
1. OpenAI GPT-4o-mini fine-tuning (scene narrative generation)
2. Local LLM fine-tuning (Llama/Mistral with LoRA)

The idea is to use your best GPT-4o + Claude cross-validated outputs as
"gold standard" training examples to train a specialized model.

Usage:
    python create_training_data.py --output-dir ./training_data
    python create_training_data.py --format openai  # For OpenAI fine-tuning
    python create_training_data.py --format alpaca  # For local LLM fine-tuning
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


def load_json(path: str) -> Optional[Dict[str, Any]]:
    """Load JSON file if exists."""
    if Path(path).exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def create_scene_analysis_prompt(paired_data: List[Dict], max_segments: int = 10) -> str:
    """
    Create the input prompt from paired_data.
    This mirrors what we send to GPT-4o in scene_analyzer.py
    """
    lines = ["Analyze this movie scene based on the following transcript and emotion data:\n"]
    
    for seg in paired_data[:max_segments]:
        emotion = seg.get("fused_emotion") or seg.get("speech_emotion") or "unknown"
        conflict = ""
        if seg.get("conflict_type"):
            conflict = f" [CONFLICT: voice={seg.get('speech_emotion')}, face={seg.get('face_emotion')}]"
        
        lines.append(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] \"{seg['text']}\"\n"
                    f"  Emotion: {emotion} (confidence: {seg.get('fused_confidence', 0):.2f}){conflict}")
    
    lines.append("\nProvide a comprehensive scene analysis including:")
    lines.append("1. Scene overview (setting, mood, atmosphere)")
    lines.append("2. Character emotional states and arcs")
    lines.append("3. Thematic observations")
    lines.append("4. Interpretation of any audio/visual emotion conflicts")
    
    return "\n".join(lines)


def create_ideal_output(
    scene_analysis: Dict,
    claude_analysis: Optional[Dict] = None,
    refined_narrative: Optional[Dict] = None
) -> str:
    """
    Create the ideal output response from existing analyses.
    Prefers refined_narrative > cross-validated consensus > single analysis.
    """
    # Best option: refined narrative
    if refined_narrative and "unified_narrative" in refined_narrative:
        narrative = refined_narrative["unified_narrative"]
        journey = refined_narrative.get("emotional_journey", {})
        themes = refined_narrative.get("thematic_synthesis", {})
        
        output = f"""## Scene Analysis

### Overview
{narrative.get('opening', '')}

### Development
{narrative.get('development', '')}

### Climax
{narrative.get('climax', '')}

### Resolution
{narrative.get('resolution', '')}

### Emotional Journey
**Arc Type:** {journey.get('arc_type', 'N/A')}
{journey.get('description', '')}

### Themes
**Primary Theme:** {themes.get('primary_theme', 'N/A')}
**Supporting Themes:** {', '.join(themes.get('supporting_themes', []))}
{themes.get('how_themes_manifest', '')}"""
        return output
    
    # Second option: scene_analysis
    if scene_analysis:
        overview = scene_analysis.get("scene_overview", {})
        themes = scene_analysis.get("themes", {})
        summary = scene_analysis.get("summary", "")
        
        output = f"""## Scene Analysis

### Overview
**Setting:** {overview.get('setting', 'N/A')}
**Mood:** {overview.get('mood', 'N/A')}
**Atmosphere:** {overview.get('atmosphere', 'N/A')}

### Summary
{summary}

### Themes
{', '.join(themes.get('central_themes', []))}
{themes.get('narrative_significance', '')}"""
        
        # Add conflict interpretations if present
        conflicts = scene_analysis.get("conflict_interpretations", [])
        if conflicts:
            output += "\n\n### Multimodal Conflict Interpretations"
            for c in conflicts:
                output += f"\n- **{c.get('timestamp', '')}**: {c.get('interpretation', '')}"
        
        return output
    
    return "Analysis not available."


def create_openai_format(
    prompt: str,
    response: str,
    system_prompt: str = "You are an expert film analyst specializing in multimodal scene interpretation. Analyze scenes considering dialogue, emotions, visual cues, and narrative structure."
) -> Dict[str, Any]:
    """
    Create OpenAI fine-tuning format (messages array).
    """
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    }


def create_alpaca_format(prompt: str, response: str) -> Dict[str, Any]:
    """
    Create Alpaca/Stanford format for local LLM fine-tuning.
    """
    return {
        "instruction": "Analyze the following movie scene based on transcript and emotion data.",
        "input": prompt,
        "output": response
    }


def create_sharegpt_format(prompt: str, response: str) -> Dict[str, Any]:
    """
    Create ShareGPT format (used by many fine-tuning frameworks).
    """
    return {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": response}
        ]
    }


def generate_training_examples(
    data_dir: str = ".",
    format_type: str = "openai"
) -> List[Dict[str, Any]]:
    """
    Generate training examples from existing analysis outputs.
    
    Args:
        data_dir: Directory containing paired_data.json, scene_analysis.json, etc.
        format_type: "openai", "alpaca", or "sharegpt"
    
    Returns:
        List of training examples in the specified format
    """
    # Load all available data
    paired_data = load_json(os.path.join(data_dir, "paired_data.json"))
    scene_analysis = load_json(os.path.join(data_dir, "scene_analysis.json"))
    claude_analysis = load_json(os.path.join(data_dir, "claude_analysis.json"))
    refined_narrative = load_json(os.path.join(data_dir, "refined_narrative.json"))
    emotion_arc = load_json(os.path.join(data_dir, "emotion_arc.json"))
    
    if not paired_data or not scene_analysis:
        print(f"Warning: Missing required data in {data_dir}")
        return []
    
    examples = []
    
    # Example 1: Full scene analysis
    prompt = create_scene_analysis_prompt(paired_data)
    response = create_ideal_output(scene_analysis, claude_analysis, refined_narrative)
    
    if format_type == "openai":
        examples.append(create_openai_format(prompt, response))
    elif format_type == "alpaca":
        examples.append(create_alpaca_format(prompt, response))
    elif format_type == "sharegpt":
        examples.append(create_sharegpt_format(prompt, response))
    
    # Example 2: Segment-level analysis (for variety)
    # Create examples for individual segments with conflicts
    conflict_segments = [s for s in paired_data if s.get("conflict_type")]
    for seg in conflict_segments[:3]:  # Limit to 3 conflict examples
        seg_prompt = f"""Analyze this movie dialogue moment:

Timestamp: {seg['start']:.1f}s - {seg['end']:.1f}s
Dialogue: "{seg['text']}"
Voice Emotion: {seg.get('speech_emotion')} (confidence: {seg.get('speech_confidence', 0):.2f})
Facial Emotion: {seg.get('face_emotion')} (confidence: {seg.get('face_confidence', 0):.2f})
Conflict Type: {seg.get('conflict_type')}

Explain why the audio and visual emotions might differ and what this reveals about the character."""

        # Find matching conflict interpretation from scene_analysis
        conflict_interps = scene_analysis.get("conflict_interpretations", [])
        matching_interp = next(
            (c for c in conflict_interps if str(seg['start']) in c.get('timestamp', '')),
            None
        )
        
        if matching_interp:
            seg_response = f"""## Multimodal Conflict Analysis

**Observed Discrepancy:** The character's voice registers as "{seg.get('speech_emotion')}" while their facial expression shows "{seg.get('face_emotion')}".

**Interpretation:** {matching_interp.get('interpretation', 'This suggests a disconnect between expressed and felt emotions.')}

**Narrative Significance:** {matching_interp.get('narrative_significance', 'This moment reveals inner conflict within the character.')}

**Possible Explanations:**
- The character may be masking their true feelings
- This could indicate sarcasm or irony
- Social context may require emotional suppression"""

            if format_type == "openai":
                examples.append(create_openai_format(seg_prompt, seg_response))
            elif format_type == "alpaca":
                examples.append(create_alpaca_format(seg_prompt, seg_response))
            elif format_type == "sharegpt":
                examples.append(create_sharegpt_format(seg_prompt, seg_response))
    
    # Example 3: Emotional arc analysis
    if emotion_arc:
        arc_summary = emotion_arc.get("summary", {})
        arc_prompt = f"""Analyze the emotional arc of this scene:

Duration: {arc_summary.get('duration_seconds', 0):.1f} seconds
Segments: {arc_summary.get('total_segments', 0)}
Dominant Emotion: {arc_summary.get('dominant_emotion', 'unknown')}
Trajectory: {arc_summary.get('trajectory', 'unknown')}

Turning Points:
{json.dumps(arc_summary.get('key_turning_points', []), indent=2)}

Describe the emotional journey and its narrative significance."""

        arc_response = f"""## Emotional Arc Analysis

**Overall Trajectory:** {arc_summary.get('trajectory_description', 'N/A')}

**Arc Phases:**
{chr(10).join(f"- **{p['phase'].title()}** ({p['time_range']}): {p['description']}" for p in arc_summary.get('phases', []))}

**Key Turning Points:**
{chr(10).join(f"- **{tp['time']}**: {tp['shift']} — {', '.join(tp['reasons'])}" for tp in arc_summary.get('key_turning_points', []))}

**Narrative Significance:**
The {arc_summary.get('trajectory', 'emotional')} trajectory creates a {arc_summary.get('dominant_emotion', 'complex')} 
atmosphere that {arc_summary.get('trajectory_description', 'drives the scene forward')}."""

        if format_type == "openai":
            examples.append(create_openai_format(arc_prompt, arc_response))
        elif format_type == "alpaca":
            examples.append(create_alpaca_format(arc_prompt, arc_response))
        elif format_type == "sharegpt":
            examples.append(create_sharegpt_format(arc_prompt, arc_response))
    
    return examples


def save_training_data(
    examples: List[Dict[str, Any]],
    output_path: str,
    format_type: str = "openai"
):
    """
    Save training data to file.
    OpenAI format uses JSONL, others use JSON.
    """
    if format_type == "openai":
        # JSONL format for OpenAI
        with open(output_path, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
    else:
        # Standard JSON for other formats
        with open(output_path, "w") as f:
            json.dump(examples, f, indent=2)
    
    print(f"Saved {len(examples)} training examples to {output_path}")


def create_validation_split(
    examples: List[Dict[str, Any]],
    val_ratio: float = 0.2
) -> tuple:
    """Split examples into train/validation sets."""
    random.shuffle(examples)
    split_idx = int(len(examples) * (1 - val_ratio))
    return examples[:split_idx], examples[split_idx:]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate training data for fine-tuning")
    parser.add_argument("--data-dir", default=".", help="Directory with analysis outputs")
    parser.add_argument("--output-dir", default="./finetune/training_data", help="Output directory")
    parser.add_argument("--format", choices=["openai", "alpaca", "sharegpt"], default="openai",
                       help="Output format for training data")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating training data from {args.data_dir}")
    print(f"Format: {args.format}")
    
    # Generate examples
    examples = generate_training_examples(args.data_dir, args.format)
    
    if not examples:
        print("No training examples generated. Make sure you have analysis outputs.")
        return
    
    print(f"Generated {len(examples)} training examples")
    
    # Split into train/validation
    train_examples, val_examples = create_validation_split(examples, args.val_split)
    
    # Determine file extension
    ext = ".jsonl" if args.format == "openai" else ".json"
    
    # Save files
    train_path = os.path.join(args.output_dir, f"train{ext}")
    val_path = os.path.join(args.output_dir, f"validation{ext}")
    
    save_training_data(train_examples, train_path, args.format)
    save_training_data(val_examples, val_path, args.format)
    
    print(f"\nTraining set: {len(train_examples)} examples")
    print(f"Validation set: {len(val_examples)} examples")
    print(f"\nFiles saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
