"""
Scene Analyzer for Phase 2 - Semantic Understanding

Uses GPT-4o Vision to analyze keyframes and generate rich narrative explanations
of the scene, including:
- Scene context (setting, location, characters)
- Mood and cinematic style
- Thematic observations
- Multimodal conflict resolution (interpreting audio/visual emotion mismatches)

Requires: OPENAI_API_KEY environment variable
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


def build_system_prompt() -> str:
    """Build the system prompt for scene analysis."""
    return """You are an expert film analyst specializing in multimodal scene interpretation. 
Your task is to analyze movie scenes using visual frames, transcripts, and emotion data.

You excel at:
1. **Scene Description**: Identifying setting, location, time of day, and visual atmosphere
2. **Character Analysis**: Understanding actor emotions, body language, and performance nuances
3. **Cinematic Style**: Recognizing lighting, framing, color grading, and directorial choices
4. **Emotional Interpretation**: Analyzing the emotional arc of a scene
5. **Multimodal Conflict Resolution**: When audio emotion (from voice) differs from visual emotion 
   (from facial expressions), you can interpret WHY this might be happening:
   - Sarcasm or irony (calm voice + angry face)
   - Suppressed emotions (neutral voice + sad face)  
   - Performance/acting choices
   - Tension between what's said and what's felt

Provide your analysis in a structured JSON format."""


def build_analysis_prompt(
    keyframes: List[Dict[str, Any]], 
    conflict_summary: Dict[str, Any]
) -> str:
    
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
    
    return f"""Analyze this movie scene based on the provided frames and data.

TRANSCRIPT WITH TIMESTAMPS:
{transcript_context}
{conflict_context}

Please analyze and provide:

1. **Scene Overview**
   - Setting and location
   - Time of day / lighting conditions
   - Visual atmosphere and mood

2. **Character Analysis**
   - Who appears in the scene
   - Their emotional states throughout
   - Body language and performance observations

3. **Cinematic Elements**
   - Camera work and framing
   - Color palette and lighting style
   - Any notable visual techniques

4. **Emotional Arc**
   - How emotions evolve through the scene
   - Key emotional beats/moments

5. **Conflict Interpretation** (if applicable)
   - For each audio/visual emotion mismatch, explain:
     * What might be causing the discrepancy
     * What this reveals about the character's inner state
     * How this contributes to the scene's meaning

6. **Thematic Observations**
   - Central themes or ideas
   - Narrative significance

Respond with valid JSON in this structure:
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


def analyze_scene(
    keyframes: List[Dict[str, Any]],
    conflict_summary: Dict[str, Any],
    max_frames: int = 10,
    model: str = "gpt-4o"
) -> Dict[str, Any]:

    if not OpenAI:
        raise ImportError("openai package not installed")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    
    # Select frames to send (prioritize high-priority and conflict frames)
    sorted_keyframes = sorted(keyframes, key=lambda x: x.get("priority", 0), reverse=True)
    selected_keyframes = sorted_keyframes[:max_frames]
    # Re-sort by timestamp for coherent analysis
    selected_keyframes = sorted(selected_keyframes, key=lambda x: x.get("start", 0))
    
    print(f"Analyzing {len(selected_keyframes)} keyframes with {model}...")
    
    # Build message content with images
    content = []
    
    # Add text prompt
    content.append({
        "type": "text",
        "text": build_analysis_prompt(selected_keyframes, conflict_summary)
    })
    
    # Add images
    for kf in selected_keyframes:
        frame_path = kf["frame"]["path"]
        if Path(frame_path).exists():
            base64_image = encode_image_base64(frame_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low"  # This reduces tokens
                }
            })
    
    # Call GPT-4o
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": content}
        ],
        max_tokens=2000,
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    # Parse response
    result_text = response.choices[0].message.content
    try:
        result = json.loads(result_text)
    except json.JSONDecodeError:
        result = {"raw_response": result_text, "parse_error": True}
    
    # Add metadata
    result["_metadata"] = {
        "model": model,
        "frames_analyzed": len(selected_keyframes),
        "total_keyframes": len(keyframes),
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    }
    
    return result


def save_analysis(analysis: Dict[str, Any], output_path: str = "scene_analysis.json"):
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved analysis to {output_path}")


def main():
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):

        
        # Load data for preview
        keyframes = load_keyframes()
        conflict_summary = load_conflict_summary()

        return
    
    # Load data
    keyframes = load_keyframes()
    
    conflict_summary = load_conflict_summary()
    
    # Run analysis
    analysis = analyze_scene(keyframes, conflict_summary)
    
    # Save results
    save_analysis(analysis)
    


if __name__ == "__main__":
    main()

