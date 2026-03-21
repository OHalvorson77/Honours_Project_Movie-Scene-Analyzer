"""
Narrative Refiner

Final pass that ensures narrative consistency across the entire analysis.
Takes outputs from:
- scene_analysis.json (GPT-4o)
- claude_analysis.json (Claude)
- contextual_analysis.json (contextual analyzer)
- emotion_arc.json (arc data)

And produces a unified, coherent narrative that:
1. Resolves inconsistencies between different analysis sources
2. Ensures the emotional arc is preserved intact
3. Creates a smooth narrative flow across all segments

This addresses the professor's feedback on maintaining "narrative consistency
and synergetic effect across multi-scene summaries."
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def load_json_safe(path: str) -> Optional[Dict[str, Any]]:
    """Load JSON file if it exists."""
    if Path(path).exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def build_refinement_prompt(
    gpt4o_analysis: Optional[Dict[str, Any]],
    claude_analysis: Optional[Dict[str, Any]],
    contextual_analysis: Optional[Dict[str, Any]],
    arc_data: Optional[Dict[str, Any]]
) -> str:
    """Build the prompt for narrative refinement."""
    
    sections = []
    
    # Arc summary
    if arc_data:
        summary = arc_data.get("summary", {})
        sections.append(f"""EMOTIONAL ARC DATA:
- Trajectory: {summary.get('trajectory', 'unknown')} ({summary.get('trajectory_description', '')})
- Dominant Emotion: {summary.get('dominant_emotion', 'unknown')}
- Conflicts Detected: {summary.get('conflicts_detected', 0)}
- Phases: {', '.join(p['phase'] for p in summary.get('phases', []))}
- Key Turning Points: {len(summary.get('key_turning_points', []))}""")
    
    # GPT-4o analysis
    if gpt4o_analysis:
        sections.append(f"""GPT-4o ANALYSIS:
- Scene Summary: {gpt4o_analysis.get('summary', 'N/A')[:300]}...
- Themes: {', '.join(gpt4o_analysis.get('themes', {}).get('central_themes', []))}
- Mood: {gpt4o_analysis.get('scene_overview', {}).get('mood', 'N/A')}""")
    
    # Claude analysis
    if claude_analysis:
        sections.append(f"""CLAUDE ANALYSIS:
- Scene Summary: {claude_analysis.get('summary', 'N/A')[:300]}...
- Themes: {', '.join(claude_analysis.get('themes', {}).get('central_themes', []))}
- Mood: {claude_analysis.get('scene_overview', {}).get('mood', 'N/A')}""")
    
    # Contextual analysis
    if contextual_analysis:
        narrative = contextual_analysis.get("narrative_synthesis", {})
        sections.append(f"""CONTEXTUAL ANALYSIS (with sliding window):
- Overall Trajectory: {narrative.get('overall_trajectory', 'N/A')}
- Key Insights: {'; '.join(narrative.get('key_insights', [])[:3])}
- Phase Narratives: {len(narrative.get('phase_narratives', {}))} phases analyzed""")
    
    combined_input = "\n\n".join(sections)
    
    return f"""You are a narrative editor specializing in film analysis. Your task is to synthesize multiple analysis sources into a SINGLE, COHERENT narrative that preserves the emotional arc.

{combined_input}

YOUR TASK:
Create a unified scene analysis that:
1. Maintains narrative consistency from beginning to end
2. Preserves the emotional arc (trajectory: {arc_data.get('summary', {}).get('trajectory', 'unknown') if arc_data else 'unknown'})
3. Resolves any contradictions between GPT-4o and Claude (prefer the more specific/grounded observation)
4. Ensures each phase flows naturally into the next
5. Highlights the key emotional turning points

OUTPUT FORMAT (JSON):
{{
  "unified_narrative": {{
    "opening": "How the scene begins (exposition)...",
    "development": "How tension/emotion builds (rising action)...",
    "climax": "The emotional peak of the scene...",
    "resolution": "How the scene concludes..."
  }},
  "emotional_journey": {{
    "arc_type": "ascending/descending/stable/complex",
    "description": "A 2-3 sentence description of the emotional journey...",
    "key_moments": [
      {{"timestamp": "Xs", "moment": "Description", "significance": "Why it matters"}}
    ]
  }},
  "character_analysis": {{
    "primary_character": {{
      "emotional_state": "...",
      "internal_conflict": "...",
      "arc_within_scene": "..."
    }}
  }},
  "thematic_synthesis": {{
    "primary_theme": "...",
    "supporting_themes": ["..."],
    "how_themes_manifest": "..."
  }},
  "technical_observations": {{
    "cinematic_choices": "...",
    "how_form_supports_content": "..."
  }},
  "multimodal_conflicts": [
    {{
      "description": "...",
      "interpretation": "What it reveals about the character/scene..."
    }}
  ],
  "coherence_notes": "Any notes on how you resolved contradictions between sources..."
}}"""


def refine_narrative(
    gpt4o_path: str = "scene_analysis.json",
    claude_path: str = "claude_analysis.json",
    contextual_path: str = "contextual_analysis.json",
    arc_path: str = "emotion_arc.json",
    output_path: str = "refined_narrative.json",
    model: str = "gpt-4o"
) -> Dict[str, Any]:
    """
    Run the narrative refinement pass.
    
    This is the final synthesis step that creates a coherent, unified analysis.
    """
    print("=" * 60)
    print("NARRATIVE REFINER")
    print("=" * 60)
    
    # Load all available analyses
    print("\nLoading analysis sources...")
    gpt4o = load_json_safe(gpt4o_path)
    claude = load_json_safe(claude_path)
    contextual = load_json_safe(contextual_path)
    arc_data = load_json_safe(arc_path)
    
    sources_available = []
    if gpt4o: sources_available.append("GPT-4o")
    if claude: sources_available.append("Claude")
    if contextual: sources_available.append("Contextual")
    if arc_data: sources_available.append("Arc Data")
    
    print(f"  Available sources: {', '.join(sources_available)}")
    
    if not any([gpt4o, claude, contextual]):
        print("\n⚠️  No analysis sources found. Run scene_analyzer.py first.")
        return {"error": "No analysis sources available"}
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set - returning merged data without refinement")
        return merge_without_llm(gpt4o, claude, contextual, arc_data)
    
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    # Build and send refinement prompt
    print("\nRefining narrative with GPT-4o...")
    prompt = build_refinement_prompt(gpt4o, claude, contextual, arc_data)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert film analyst and narrative editor."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    result_text = response.choices[0].message.content
    
    try:
        refined = json.loads(result_text)
    except json.JSONDecodeError:
        refined = {"raw_response": result_text, "parse_error": True}
    
    # Add metadata
    refined["_metadata"] = {
        "sources_used": sources_available,
        "model": model,
        "tokens_used": response.usage.total_tokens
    }
    
    # Add original arc data for reference
    if arc_data:
        refined["_arc_reference"] = arc_data.get("summary", {})
    
    # Save result
    with open(output_path, "w") as f:
        json.dump(refined, f, indent=2)
    print(f"\nSaved refined narrative to {output_path}")
    
    return refined


def merge_without_llm(
    gpt4o: Optional[Dict],
    claude: Optional[Dict],
    contextual: Optional[Dict],
    arc_data: Optional[Dict]
) -> Dict[str, Any]:
    """
    Merge analyses without LLM refinement (fallback when no API key).
    """
    result = {
        "merged_analysis": {},
        "sources": {}
    }
    
    if gpt4o:
        result["sources"]["gpt4o"] = {
            "summary": gpt4o.get("summary", ""),
            "themes": gpt4o.get("themes", {}),
            "scene_overview": gpt4o.get("scene_overview", {})
        }
    
    if claude:
        result["sources"]["claude"] = {
            "summary": claude.get("summary", ""),
            "themes": claude.get("themes", {}),
            "scene_overview": claude.get("scene_overview", {})
        }
    
    if contextual:
        result["sources"]["contextual"] = contextual.get("narrative_synthesis", {})
    
    if arc_data:
        result["arc_summary"] = arc_data.get("summary", {})
    
    # Simple merge: prefer GPT-4o, fill gaps with Claude
    merged = {}
    
    if gpt4o:
        merged["summary"] = gpt4o.get("summary", "")
        merged["themes"] = gpt4o.get("themes", {}).get("central_themes", [])
        merged["scene_overview"] = gpt4o.get("scene_overview", {})
    
    if claude and not merged.get("summary"):
        merged["summary"] = claude.get("summary", "")
        merged["themes"] = claude.get("themes", {}).get("central_themes", [])
        merged["scene_overview"] = claude.get("scene_overview", {})
    
    # Add arc trajectory
    if arc_data:
        summary = arc_data.get("summary", {})
        merged["emotional_arc"] = {
            "trajectory": summary.get("trajectory", ""),
            "description": summary.get("trajectory_description", ""),
            "dominant_emotion": summary.get("dominant_emotion", ""),
            "phases": summary.get("phases", [])
        }
    
    result["merged_analysis"] = merged
    result["_note"] = "Merged without LLM refinement (no API key). Run with OPENAI_API_KEY for full refinement."
    
    return result


def print_refined_summary(refined: Dict[str, Any]):
    """Print a summary of the refined narrative."""
    print("\n" + "=" * 60)
    print("REFINED NARRATIVE SUMMARY")
    print("=" * 60)
    
    if "error" in refined:
        print(f"\nError: {refined['error']}")
        return
    
    narrative = refined.get("unified_narrative", {})
    journey = refined.get("emotional_journey", {})
    themes = refined.get("thematic_synthesis", {})
    
    print(f"\n📖 NARRATIVE ARC:")
    if narrative.get("opening"):
        print(f"  Opening: {narrative['opening'][:100]}...")
    if narrative.get("climax"):
        print(f"  Climax: {narrative['climax'][:100]}...")
    if narrative.get("resolution"):
        print(f"  Resolution: {narrative['resolution'][:100]}...")
    
    print(f"\n💫 EMOTIONAL JOURNEY:")
    print(f"  Arc Type: {journey.get('arc_type', 'N/A')}")
    print(f"  Description: {journey.get('description', 'N/A')}")
    
    print(f"\n🎭 THEMES:")
    print(f"  Primary: {themes.get('primary_theme', 'N/A')}")
    if themes.get("supporting_themes"):
        print(f"  Supporting: {', '.join(themes['supporting_themes'])}")
    
    if refined.get("multimodal_conflicts"):
        print(f"\n⚡ CONFLICT INTERPRETATIONS:")
        for conflict in refined["multimodal_conflicts"][:2]:
            print(f"  - {conflict.get('description', 'N/A')}")
            print(f"    Interpretation: {conflict.get('interpretation', 'N/A')[:80]}...")
    
    if refined.get("coherence_notes"):
        print(f"\n📝 COHERENCE NOTES:")
        print(f"  {refined['coherence_notes'][:200]}...")


def main():
    """Run the narrative refiner."""
    refined = refine_narrative()
    print_refined_summary(refined)


if __name__ == "__main__":
    main()
