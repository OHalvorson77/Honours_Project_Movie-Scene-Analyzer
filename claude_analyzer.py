"""
Claude Scene Analyzer for Phase 2 - Cross-Validation

Uses Claude 3.5 Sonnet to analyze keyframes and generate narrative explanations,
providing a second opinion for comparison with GPT-4o results.

Claude often excels at:
- Coherent long-form generation
- Nuanced emotional interpretation
- Safety-aware analysis

Requires: ANTHROPIC_API_KEY environment variable
"""

import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Any

try:
    import anthropic
except ImportError:
    print("Please install anthropic: pip install anthropic")
    anthropic = None


def encode_image_base64(image_path: str) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def load_keyframes(path: str = "keyframes.json") -> List[Dict[str, Any]]:
    """Load extracted keyframes."""
    with open(path, "r") as f:
        return json.load(f)


def load_conflict_summary(path: str = "conflict_summary.json") -> Dict[str, Any]:
    """Load the conflict summary for context."""
    with open(path, "r") as f:
        return json.load(f)


def build_analysis_prompt(
    keyframes: List[Dict[str, Any]], 
    conflict_summary: Dict[str, Any]
) -> str:
    """Build the analysis prompt with transcript and emotion context."""
    
    # Build transcript context
    transcript_lines = []
    for kf in keyframes:
        emotion_note = ""
        if kf.get("conflict_type"):
            emotion_note = f" [CONFLICT: voice={kf['speech_emotion']}, face={kf['face_emotion']}]"
        transcript_lines.append(
            f"[{kf['start']:.1f}s] \"{kf['text']}\"{emotion_note}"
        )
    
    transcript_context = "\n".join(transcript_lines)
    
    # Build conflict context
    conflict_context = ""
    if conflict_summary.get("conflicts"):
        conflict_context = "\n\nEMOTION CONFLICTS DETECTED:\n"
        for c in conflict_summary["conflicts"]:
            conflict_context += f"- At {c['time']}: Voice shows '{c['speech_emotion']}' but face shows '{c['face_emotion']}'\n"
            conflict_context += f"  Text: \"{c['text']}\"\n"
    
    return f"""You are an expert film analyst. Analyze this movie scene based on the provided frames and emotion data.

TRANSCRIPT WITH TIMESTAMPS:
{transcript_context}
{conflict_context}

Provide a comprehensive analysis covering:

1. **Scene Overview**: Setting, location, time of day, visual atmosphere, and mood
2. **Character Analysis**: Who appears, their emotional states, body language, performance observations
3. **Cinematic Elements**: Camera work, lighting, color palette, notable visual techniques
4. **Emotional Arc**: How emotions evolve, key emotional beats
5. **Conflict Interpretation**: For each audio/visual emotion mismatch, explain what might cause it and what it reveals about the character
6. **Themes**: Central themes and narrative significance

Respond with valid JSON in this exact structure:
{{
  "scene_overview": {{
    "setting": "...",
    "time_of_day": "...",
    "atmosphere": "...",
    "mood": "..."
  }},
  "characters": [
    {{
      "description": "...",
      "emotional_journey": "...",
      "performance_notes": "..."
    }}
  ],
  "cinematic_elements": {{
    "camera_work": "...",
    "lighting": "...",
    "color_palette": "...",
    "notable_techniques": "..."
  }},
  "emotional_arc": {{
    "progression": "...",
    "key_moments": ["..."]
  }},
  "conflict_interpretations": [
    {{
      "timestamp": "...",
      "speech_emotion": "...",
      "face_emotion": "...",
      "interpretation": "...",
      "narrative_significance": "..."
    }}
  ],
  "themes": {{
    "central_themes": ["..."],
    "narrative_significance": "..."
  }},
  "summary": "A 2-3 sentence overall summary of the scene"
}}"""


def analyze_scene_claude(
    keyframes: List[Dict[str, Any]],
    conflict_summary: Dict[str, Any],
    max_frames: int = 10,
    model: str = "claude-sonnet-4-20250514"
) -> Dict[str, Any]:
    """
    Analyze a scene using Claude 3.5 Sonnet.
    
    Args:
        keyframes: List of keyframe objects with frame paths
        conflict_summary: Summary of emotion conflicts
        max_frames: Maximum number of frames to send
        model: Anthropic model to use
    
    Returns:
        Analysis results as a dictionary
    """
    if not anthropic:
        raise ImportError("anthropic package not installed")
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Select frames to send (prioritize high-priority and conflict frames)
    sorted_keyframes = sorted(keyframes, key=lambda x: x.get("priority", 0), reverse=True)
    selected_keyframes = sorted_keyframes[:max_frames]
    # Re-sort by timestamp for coherent analysis
    selected_keyframes = sorted(selected_keyframes, key=lambda x: x.get("start", 0))
    
    print(f"Analyzing {len(selected_keyframes)} keyframes with Claude {model}...")
    
    # Build message content with images
    content = []
    
    # Add images first (Claude prefers images before text)
    for kf in selected_keyframes:
        frame_path = kf["frame"]["path"]
        if Path(frame_path).exists():
            base64_image = encode_image_base64(frame_path)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_image
                }
            })
    
    # Add text prompt
    content.append({
        "type": "text",
        "text": build_analysis_prompt(selected_keyframes, conflict_summary)
    })
    
    # Call Claude
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[
            {"role": "user", "content": content}
        ]
    )
    
    # Parse response
    result_text = response.content[0].text
    try:
        result = json.loads(result_text)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\{[\s\S]*\}', result_text)
        if json_match:
            try:
                result = json.loads(json_match.group())
            except json.JSONDecodeError:
                result = {"raw_response": result_text, "parse_error": True}
        else:
            result = {"raw_response": result_text, "parse_error": True}
    
    # Add metadata
    result["_metadata"] = {
        "model": model,
        "frames_analyzed": len(selected_keyframes),
        "total_keyframes": len(keyframes),
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens
        }
    }
    
    return result


def save_analysis(analysis: Dict[str, Any], output_path: str = "claude_analysis.json"):
    """Save the scene analysis to JSON."""
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved Claude analysis to {output_path}")


def main():
    """Run the Claude scene analyzer."""
    print("=" * 60)
    print("Claude Scene Analyzer - Phase 2 Cross-Validation")
    print("=" * 60)
    
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n⚠️  ANTHROPIC_API_KEY not set!")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        print("\nGenerating analysis preview without API call...\n")
        
        # Load data for preview
        keyframes = load_keyframes()
        conflict_summary = load_conflict_summary()
        
        print(f"Would analyze {len(keyframes)} keyframes")
        print(f"Found {conflict_summary['conflict_count']} emotion conflicts")
        print("\nSample prompt preview:")
        print("-" * 40)
        print(build_analysis_prompt(keyframes[:3], conflict_summary)[:500] + "...")
        return
    
    # Load data
    print("\nLoading keyframes...")
    keyframes = load_keyframes()
    
    print("Loading conflict summary...")
    conflict_summary = load_conflict_summary()
    
    # Run analysis
    print("\nCalling Claude 3.5 Sonnet...")
    analysis = analyze_scene_claude(keyframes, conflict_summary)
    
    # Save results
    save_analysis(analysis)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CLAUDE ANALYSIS COMPLETE")
    print("=" * 60)
    
    if "summary" in analysis:
        print(f"\nScene Summary:\n{analysis['summary']}")
    
    if "themes" in analysis and "central_themes" in analysis["themes"]:
        print(f"\nThemes: {', '.join(analysis['themes']['central_themes'])}")
    
    if "_metadata" in analysis:
        meta = analysis["_metadata"]
        print(f"\nTokens used: {meta['usage']['input_tokens'] + meta['usage']['output_tokens']}")
    
    print(f"\nFull analysis saved to claude_analysis.json")


if __name__ == "__main__":
    main()



