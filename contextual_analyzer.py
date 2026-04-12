"""
Contextual Scene Analyzer

Enhanced scene analysis with sliding context window that provides:
1. Previous segment context to maintain narrative continuity
2. Emotional arc information for grounding
3. Scene phase awareness (exposition, climax, resolution)

This addresses the professor's feedback on "contextual grounding" and ensuring
the model can bridge "what is said" vs "what is happening" with narrative consistency.
"""

import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    OpenAI = None

from emotion_arc import get_context_for_segment, load_paired_data


def encode_image_base64(image_path: str) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def load_json(path: str) -> Any:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def build_contextual_system_prompt() -> str:
    return """You are an expert film analyst specializing in multimodal scene interpretation with a focus on NARRATIVE CONTINUITY.

Your key responsibilities:
1. **Contextual Grounding**: Bridge the gap between "what is said" (dialogue) and "what is happening" (visual/emotional context)
2. **Narrative Consistency**: Ensure your analysis connects to previous segments and maintains story coherence
3. **Emotional Arc Awareness**: Understand where this moment fits in the overall emotional journey
4. **Multimodal Conflict Resolution**: Interpret discrepancies between audio and visual emotional signals

When analyzing segments:
- Reference previous context when relevant ("Following the earlier tension...", "Building on the character's previous sadness...")
- Note how emotions evolve from previous segments
- Identify what visual cues support or contradict the dialogue
- Consider the narrative phase (exposition, rising action, climax, etc.)

Provide analysis that reads as part of a coherent narrative, not isolated observations."""


def build_segment_context(
    current_idx: int,
    segments: List[Dict[str, Any]],
    arc_data: Dict[str, Any],
    window_size: int = 3
) -> str:
    """
    Build context from previous segments for narrative continuity.
    
    This creates a "sliding window" of context so GPT-4o understands
    what happened before the current segment.
    """
    context_parts = []
    
    # Get arc context (phase, trajectory, etc.)
    arc_context = get_context_for_segment(arc_data, current_idx)
    
    context_parts.append(f"""NARRATIVE CONTEXT:
- Scene Progress: {arc_context['scene_progress']*100:.0f}% through the clip
- Current Phase: {arc_context['current_phase'].upper()}
- Phase Description: {arc_context['phase_description']}
- Overall Trajectory: {arc_context['overall_trajectory']}
- Scene's Dominant Emotion: {arc_context['dominant_emotion']}""")
    
    if arc_context['is_near_turning_point']:
        context_parts.append("\n⚡ NOTE: This segment is near a significant emotional turning point!")
    
    # Get previous segments for context
    start_idx = max(0, current_idx - window_size)
    if start_idx < current_idx:
        context_parts.append("\nPREVIOUS CONTEXT (what happened before):")
        for i in range(start_idx, current_idx):
            seg = segments[i]
            emotion = seg.get("fused_emotion") or seg.get("speech_emotion") or "unknown"
            conflict = f" [CONFLICT: {seg['conflict_type']}]" if seg.get("conflict_type") else ""
            context_parts.append(
                f"  [{seg['start']:.1f}s] \"{seg['text'][:60]}...\" — Emotion: {emotion}{conflict}"
            )
    
    return "\n".join(context_parts)


def build_contextual_prompt(
    segment: Dict[str, Any],
    context: str,
    conflict_info: Optional[Dict[str, Any]] = None
) -> str:
    """Build analysis prompt with contextual grounding."""
    
    conflict_section = ""
    if segment.get("conflict_type"):
        conflict_section = f"""
MULTIMODAL CONFLICT DETECTED:
- Voice emotion: {segment.get('speech_emotion')}
- Facial emotion: {segment.get('face_emotion')}
- Conflict type: {segment.get('conflict_type')}

Please interpret this discrepancy. What might explain why the audio and visual emotions differ?
Consider: sarcasm, suppressed emotions, social masking, performance choices, or narrative subtext."""

    return f"""{context}

CURRENT SEGMENT TO ANALYZE:
- Timestamp: {segment.get('start', 0):.1f}s - {segment.get('end', 0):.1f}s
- Dialogue: "{segment.get('text', '')}"
- Speech Emotion: {segment.get('speech_emotion')} (confidence: {segment.get('speech_confidence', 0):.2f})
- Facial Emotion: {segment.get('face_emotion')} (confidence: {segment.get('face_confidence', 0):.2f})
- Fused Emotion: {segment.get('fused_emotion')}
{conflict_section}

ANALYSIS INSTRUCTIONS:
1. **Visual Grounding**: Describe what you see in the frame that supports or contradicts the emotion data
2. **Narrative Connection**: How does this moment connect to what came before?
3. **Subtext**: What's happening beneath the surface? (emotional subtext, character psychology)
4. **Significance**: Why does this moment matter in the overall scene?

Respond with valid JSON:
{{
  "visual_description": "What you observe in the frame...",
  "emotional_state": {{
    "primary_emotion": "...",
    "supporting_evidence": "Visual/audio cues that support this...",
    "subtext": "What the character might really be feeling..."
  }},
  "narrative_connection": {{
    "builds_on": "How this connects to previous moments...",
    "foreshadows": "What this might be setting up..."
  }},
  "conflict_interpretation": "If conflict exists, explain it...",
  "significance": "Why this moment matters..."
}}"""


def analyze_with_context(
    segments: List[Dict[str, Any]],
    arc_data: Dict[str, Any],
    keyframes: List[Dict[str, Any]],
    max_frames: int = 10,
    model: str = "gpt-4o"
) -> Dict[str, Any]:

    if not OpenAI:
        raise ImportError("openai package not installed")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    
    # Build segment index lookup from keyframes
    keyframe_by_segment = {}
    for kf in keyframes:
        seg_idx = kf.get("segment_index")
        if seg_idx not in keyframe_by_segment:
            keyframe_by_segment[seg_idx] = kf
    
    # Select which segments to analyze (prioritize keyframes)
    segments_to_analyze = sorted(keyframe_by_segment.keys())[:max_frames]
    
    print(f"Analyzing {len(segments_to_analyze)} segments with contextual grounding...")
    
    segment_analyses = []
    accumulated_context = []  # Build up context as we go
    
    for seg_idx in segments_to_analyze:
        segment = segments[seg_idx]
        keyframe = keyframe_by_segment[seg_idx]
        
        print(f"  Analyzing segment {seg_idx} [{segment.get('start', 0):.1f}s]...")
        
        # Build context string
        context = build_segment_context(seg_idx, segments, arc_data)
        
        # Add accumulated analysis context (what we've learned so far)
        if accumulated_context:
            context += "\n\nPREVIOUS ANALYSIS INSIGHTS:\n"
            for ac in accumulated_context[-3:]:  # Last 3 analyses
                context += f"  [{ac['timestamp']}] {ac['insight']}\n"
        
        # Build prompt
        prompt = build_contextual_prompt(segment, context)
        
        # Prepare image
        content = [{"type": "text", "text": prompt}]
        
        frame_path = keyframe["frame"]["path"]
        if Path(frame_path).exists():
            base64_image = encode_image_base64(frame_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low"
                }
            })
        
        # Call GPT-4o
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": build_contextual_system_prompt()},
                    {"role": "user", "content": content}
                ],
                max_tokens=800,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            # Add metadata
            result["_segment"] = {
                "index": seg_idx,
                "timestamp": f"{segment.get('start', 0):.1f}s",
                "text": segment.get("text", ""),
                "fused_emotion": segment.get("fused_emotion"),
                "conflict_type": segment.get("conflict_type")
            }
            
            segment_analyses.append(result)
            
            # Add to accumulated context
            insight = result.get("significance", result.get("emotional_state", {}).get("primary_emotion", ""))
            accumulated_context.append({
                "timestamp": f"{segment.get('start', 0):.1f}s",
                "insight": insight[:100] if insight else "Analyzed"
            })
            
        except Exception as e:
            print(f"    Error analyzing segment {seg_idx}: {e}")
            segment_analyses.append({
                "error": str(e),
                "_segment": {"index": seg_idx, "timestamp": f"{segment.get('start', 0):.1f}s"}
            })
    
    return {
        "segment_analyses": segment_analyses,
        "arc_summary": arc_data.get("summary", {}),
        "_metadata": {
            "model": model,
            "segments_analyzed": len(segment_analyses),
            "context_window_used": True
        }
    }


def synthesize_narrative(
    contextual_analysis: Dict[str, Any],
    arc_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Synthesize individual segment analyses into a coherent narrative.
    
    This creates the final unified analysis that maintains narrative consistency
    across all analyzed segments.
    """
    segment_analyses = contextual_analysis.get("segment_analyses", [])
    arc_summary = arc_data.get("summary", {})
    phases = arc_data.get("phases", [])
    
    # Group analyses by phase
    phase_narratives = {}
    for phase in phases:
        phase_name = phase.get("phase", "unknown")
        phase_start = phase.get("start_segment", 0)
        phase_end = phase.get("end_segment", 999)
        
        phase_analyses = [
            a for a in segment_analyses
            if phase_start <= a.get("_segment", {}).get("index", -1) <= phase_end
        ]
        
        if phase_analyses:
            phase_narratives[phase_name] = {
                "time_range": f"{phase.get('start_time', 0):.1f}s - {phase.get('end_time', 0):.1f}s",
                "description": phase.get("description", ""),
                "key_moments": [
                    {
                        "timestamp": a["_segment"]["timestamp"],
                        "dialogue": a["_segment"]["text"][:50] + "..." if len(a["_segment"].get("text", "")) > 50 else a["_segment"].get("text", ""),
                        "visual": a.get("visual_description", ""),
                        "significance": a.get("significance", "")
                    }
                    for a in phase_analyses
                ]
            }
    
    # Build narrative thread
    narrative_thread = []
    for analysis in segment_analyses:
        connection = analysis.get("narrative_connection", {})
        if connection.get("builds_on"):
            narrative_thread.append({
                "timestamp": analysis["_segment"]["timestamp"],
                "builds_on": connection["builds_on"],
                "foreshadows": connection.get("foreshadows", "")
            })
    
    # Collect conflict interpretations
    conflicts = [
        {
            "timestamp": a["_segment"]["timestamp"],
            "conflict_type": a["_segment"].get("conflict_type"),
            "interpretation": a.get("conflict_interpretation", "")
        }
        for a in segment_analyses
        if a.get("_segment", {}).get("conflict_type")
    ]
    
    return {
        "overall_trajectory": arc_summary.get("trajectory_description", ""),
        "dominant_emotion": arc_summary.get("dominant_emotion", ""),
        "phase_narratives": phase_narratives,
        "narrative_thread": narrative_thread,
        "conflict_interpretations": conflicts,
        "key_insights": [
            a.get("significance", "")
            for a in segment_analyses
            if a.get("significance")
        ][:5]
    }

# This function gets called from the pipeline
def run_contextual_analysis(
    paired_data_path: str = "paired_data.json",
    arc_data_path: str = "emotion_arc.json",
    keyframes_path: str = "keyframes.json",
    output_path: str = "contextual_analysis.json",
    max_frames: int = 10
) -> Dict[str, Any]:

    print("=" * 60)
    print("CONTEXTUAL SCENE ANALYZER")
    print("=" * 60)
    
    # Loading in the paired data
    print("\nLoading data...")
    segments = load_paired_data(paired_data_path)
    arc_data = load_json(arc_data_path)
    keyframes = load_json(keyframes_path)
    
    print(f"  Segments: {len(segments)}")
    print(f"  Keyframes: {len(keyframes)}")
    print(f"  Arc phases: {len(arc_data.get('phases', []))}")
    
    # Run contextual analysis
    print("\nRunning contextual analysis...")
    contextual_result = analyze_with_context(
        segments, arc_data, keyframes, max_frames=max_frames
    )
    
    # Synthesize narrative
    print("\nSynthesizing narrative...")
    narrative = synthesize_narrative(contextual_result, arc_data)
    
    # Combine results
    result = {
        "narrative_synthesis": narrative,
        "contextual_analysis": contextual_result,
        "arc_data": arc_data.get("summary", {})
    }
    
    # Save
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output_path}")
    
    return result


def main():
    """Run contextual analysis."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("\nShowing what would be analyzed...\n")
        
        # Preview mode
        segments = load_paired_data()
        arc_data = load_json("emotion_arc.json")
        
        for i in [0, len(segments)//2, len(segments)-1]:
            if i < len(segments):
                context = build_segment_context(i, segments, arc_data)
                print(f"--- Segment {i} Context ---")
                print(context[:500])
                print("...\n")
        return
    
    run_contextual_analysis()


if __name__ == "__main__":
    main()
