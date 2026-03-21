"""
LLM-as-Judge Evaluation System

Automated, reproducible narrative coherence evaluation using GPT-4 as a judge.

Evaluates generated analyses against specific criteria:
1. Logical Flow - Does the narrative progress logically?
2. Temporal Consistency - Are events described in correct order?
3. Emotion Arc Coherence - Does the emotional journey make sense?
4. Grounding - Does it reference visual/audio cues appropriately?

The judge receives:
- The generated analysis
- The source transcript with emotion data
- A detailed rubric for scoring each dimension (1-5)

Usage:
    python llm_judge.py                    # Evaluate with GPT-4
    python llm_judge.py --model claude     # Use Claude instead
    python llm_judge.py --validate         # Compare with human labels
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"
LABELS_DIR = Path(__file__).parent / "labels"

# Paths to analysis files
SCENE_ANALYSIS_PATH = Path(__file__).parent.parent / "scene_analysis.json"
PAIRED_DATA_PATH = Path(__file__).parent.parent / "paired_data.json"
EMOTION_ARC_PATH = Path(__file__).parent.parent / "emotion_arc.json"
REFINED_NARRATIVE_PATH = Path(__file__).parent.parent / "refined_narrative.json"


# ============================================================================
# Evaluation Rubric
# ============================================================================

EVALUATION_RUBRIC = """
## Narrative Coherence Evaluation Rubric

### 1. Logical Flow (1-5)
How well does the analysis progress logically from one point to the next?

1 = Disjointed, random observations with no connection
2 = Some logical connections but major gaps or contradictions
3 = Generally logical but with occasional awkward transitions
4 = Clear logical progression with minor issues
5 = Excellent flow, each point builds naturally on the previous

### 2. Temporal Consistency (1-5)
Are events and emotions described in the correct chronological order?

1 = Events described out of order, confusing timeline
2 = Some temporal errors or inconsistencies
3 = Mostly correct order with minor slips
4 = Accurate timeline with good temporal markers
5 = Perfect temporal consistency, clear progression through scene

### 3. Emotion Arc Coherence (1-5)
Does the described emotional journey align with the emotion data and make narrative sense?

1 = Emotional descriptions contradict the data or are nonsensical
2 = Significant mismatches between described and detected emotions
3 = Generally aligned but misses key emotional beats
4 = Good alignment, captures major emotional shifts
5 = Excellent, accurately captures the full emotional arc with insightful interpretation

### 4. Visual/Audio Grounding (1-5)
Does the analysis appropriately reference and interpret visual and audio cues?

1 = No reference to actual content, purely generic
2 = Vague references, doesn't connect to specific moments
3 = Some specific references but could be more grounded
4 = Good use of specific visual/audio details
5 = Excellent grounding, analysis is clearly tied to specific observable evidence

### 5. Conflict Interpretation (1-5)
If audio/visual emotion conflicts exist, are they interpreted meaningfully?

1 = Conflicts ignored or misinterpreted
2 = Conflicts acknowledged but poorly explained
3 = Basic interpretation without depth
4 = Good interpretation with plausible explanations
5 = Insightful interpretation that adds narrative value
(N/A if no conflicts present)

### 6. Overall Coherence (1-5)
Holistic assessment of the narrative quality.

1 = Incoherent, would not be useful to a reader
2 = Below average, significant improvements needed
3 = Acceptable, conveys basic information
4 = Good quality, informative and well-structured
5 = Excellent, publication-ready analysis
"""


def load_json(path: Path) -> Optional[Any]:
    """Load JSON file if exists."""
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_json(data: Any, path: Path):
    """Save data to JSON."""
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def build_judge_prompt(
    analysis: Dict[str, Any],
    transcript_data: List[Dict[str, Any]],
    emotion_arc: Optional[Dict[str, Any]] = None
) -> str:
    """
    Build the prompt for the LLM judge.
    
    Includes:
    1. The rubric
    2. Source transcript with emotion data
    3. The generated analysis to evaluate
    """
    
    # Format transcript with emotions
    transcript_text = "## Source Transcript with Emotion Data\n\n"
    for seg in transcript_data[:20]:  # Limit for context window
        emotion = seg.get("fused_emotion") or seg.get("speech_emotion") or "unknown"
        conflict = ""
        if seg.get("conflict_type"):
            conflict = f" ⚠️ CONFLICT: voice={seg.get('speech_emotion')}, face={seg.get('face_emotion')}"
        transcript_text += f"[{seg['start']:.1f}s] \"{seg['text']}\"\n"
        transcript_text += f"    Emotion: {emotion} (confidence: {seg.get('fused_confidence', 0):.2f}){conflict}\n\n"
    
    # Format emotion arc if available
    arc_text = ""
    if emotion_arc and "summary" in emotion_arc:
        summary = emotion_arc["summary"]
        arc_text = f"""
## Detected Emotion Arc

- Trajectory: {summary.get('trajectory', 'unknown')} ({summary.get('trajectory_description', '')})
- Dominant Emotion: {summary.get('dominant_emotion', 'unknown')}
- Conflicts Detected: {summary.get('conflicts_detected', 0)}
- Phases: {', '.join(p['phase'] for p in summary.get('phases', []))}
"""
    
    # Format the analysis to evaluate
    analysis_text = "## Generated Analysis to Evaluate\n\n"
    if isinstance(analysis, dict):
        if "summary" in analysis:
            analysis_text += f"**Summary:** {analysis['summary']}\n\n"
        if "scene_overview" in analysis:
            overview = analysis["scene_overview"]
            analysis_text += f"**Scene Overview:**\n"
            analysis_text += f"- Setting: {overview.get('setting', 'N/A')}\n"
            analysis_text += f"- Mood: {overview.get('mood', 'N/A')}\n"
            analysis_text += f"- Atmosphere: {overview.get('atmosphere', 'N/A')}\n\n"
        if "emotional_arc" in analysis:
            arc = analysis["emotional_arc"]
            analysis_text += f"**Emotional Arc:** {arc.get('progression', 'N/A')}\n\n"
        if "themes" in analysis:
            themes = analysis["themes"]
            analysis_text += f"**Themes:** {', '.join(themes.get('central_themes', []))}\n\n"
        if "conflict_interpretations" in analysis:
            analysis_text += f"**Conflict Interpretations:** {len(analysis['conflict_interpretations'])} provided\n"
            for ci in analysis.get("conflict_interpretations", [])[:3]:
                analysis_text += f"  - {ci.get('timestamp', 'N/A')}: {ci.get('interpretation', 'N/A')[:100]}...\n"
    else:
        analysis_text += str(analysis)[:2000]
    
    # Build full prompt
    prompt = f"""You are an expert evaluator assessing the quality of AI-generated movie scene analysis.

{EVALUATION_RUBRIC}

---

{transcript_text}

{arc_text}

---

{analysis_text}

---

## Your Task

Evaluate the generated analysis against the rubric above. For each criterion, provide:
1. A score (1-5)
2. A brief justification (1-2 sentences)

Also identify:
- Key strengths of the analysis
- Key weaknesses or areas for improvement
- Whether the analysis would be useful to someone trying to understand this scene

Respond in this exact JSON format:
{{
    "scores": {{
        "logical_flow": {{"score": <1-5>, "justification": "..."}},
        "temporal_consistency": {{"score": <1-5>, "justification": "..."}},
        "emotion_arc_coherence": {{"score": <1-5>, "justification": "..."}},
        "visual_audio_grounding": {{"score": <1-5>, "justification": "..."}},
        "conflict_interpretation": {{"score": <1-5 or null if N/A>, "justification": "..."}},
        "overall_coherence": {{"score": <1-5>, "justification": "..."}}
    }},
    "strengths": ["...", "..."],
    "weaknesses": ["...", "..."],
    "useful_to_reader": true/false,
    "summary": "2-3 sentence overall assessment"
}}

Respond ONLY with the JSON, no other text."""

    return prompt


def evaluate_with_gpt4(prompt: str) -> Dict[str, Any]:
    """Run evaluation using GPT-4."""
    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "openai package not installed"}
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}
    
    client = OpenAI(api_key=api_key)
    
    print("Calling GPT-4 as judge...")
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert evaluator of AI-generated content. Be fair but rigorous in your assessment."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.3,  # Lower temperature for more consistent evaluation
        response_format={"type": "json_object"}
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        result["_metadata"] = {
            "model": "gpt-4o",
            "tokens_used": response.usage.total_tokens
        }
        return result
    except json.JSONDecodeError:
        return {"error": "Failed to parse response", "raw": response.choices[0].message.content}


def evaluate_with_claude(prompt: str) -> Dict[str, Any]:
    """Run evaluation using Claude."""
    try:
        from anthropic import Anthropic
    except ImportError:
        return {"error": "anthropic package not installed"}
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not set"}
    
    client = Anthropic(api_key=api_key)
    
    print("Calling Claude as judge...")
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    try:
        result = json.loads(response.content[0].text)
        result["_metadata"] = {
            "model": "claude-3-5-sonnet",
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens
        }
        return result
    except json.JSONDecodeError:
        return {"error": "Failed to parse response", "raw": response.content[0].text}


def calculate_aggregate_score(evaluation: Dict[str, Any]) -> float:
    """Calculate weighted aggregate score from evaluation."""
    weights = {
        "logical_flow": 0.20,
        "temporal_consistency": 0.15,
        "emotion_arc_coherence": 0.25,
        "visual_audio_grounding": 0.20,
        "conflict_interpretation": 0.10,
        "overall_coherence": 0.10
    }
    
    scores = evaluation.get("scores", {})
    
    total = 0
    weight_sum = 0
    
    for criterion, weight in weights.items():
        score_data = scores.get(criterion, {})
        score = score_data.get("score")
        
        if score is not None and isinstance(score, (int, float)):
            total += score * weight
            weight_sum += weight
    
    return round(total / weight_sum, 2) if weight_sum > 0 else 0


def validate_against_human(
    llm_evaluation: Dict[str, Any],
    human_labels: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare LLM judge scores with human evaluation.
    
    Returns correlation metrics to validate the LLM judge.
    """
    if not human_labels or "ratings" not in human_labels:
        return {"error": "No human labels available for validation"}
    
    human_ratings = human_labels["ratings"]
    llm_scores = llm_evaluation.get("scores", {})
    
    # Map human rating keys to LLM score keys
    mapping = {
        "coherence": "logical_flow",
        "accuracy": "visual_audio_grounding",
        "insight": "emotion_arc_coherence",
        "emotion_accuracy": "emotion_arc_coherence",
        "conflict_interpretation": "conflict_interpretation"
    }
    
    comparisons = []
    for human_key, llm_key in mapping.items():
        if human_key in human_ratings and llm_key in llm_scores:
            human_score = human_ratings[human_key]
            llm_score = llm_scores[llm_key].get("score")
            
            if llm_score is not None:
                diff = abs(human_score - llm_score)
                comparisons.append({
                    "criterion": human_key,
                    "human_score": human_score,
                    "llm_score": llm_score,
                    "difference": diff,
                    "agreement": diff <= 1  # Within 1 point = agreement
                })
    
    if not comparisons:
        return {"error": "No comparable scores found"}
    
    agreement_rate = sum(1 for c in comparisons if c["agreement"]) / len(comparisons)
    avg_difference = sum(c["difference"] for c in comparisons) / len(comparisons)
    
    return {
        "comparisons": comparisons,
        "agreement_rate": round(agreement_rate, 2),
        "average_difference": round(avg_difference, 2),
        "interpretation": "Good correlation" if agreement_rate >= 0.7 else "Moderate correlation" if agreement_rate >= 0.5 else "Poor correlation"
    }


def run_llm_judge(
    model: str = "gpt4",
    validate: bool = False,
    output_path: str = None
) -> Dict[str, Any]:
    """
    Run the LLM-as-judge evaluation.
    
    Args:
        model: "gpt4" or "claude"
        validate: Compare with human labels if available
        output_path: Where to save results
    """
    print("=" * 60)
    print("LLM-AS-JUDGE EVALUATION")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    analysis = load_json(SCENE_ANALYSIS_PATH)
    transcript = load_json(PAIRED_DATA_PATH)
    emotion_arc = load_json(EMOTION_ARC_PATH)
    
    if not analysis:
        print("❌ Error: scene_analysis.json not found")
        return {"error": "No analysis to evaluate"}
    
    if not transcript:
        print("❌ Error: paired_data.json not found")
        return {"error": "No transcript data"}
    
    print(f"  Analysis: ✓")
    print(f"  Transcript: {len(transcript)} segments")
    print(f"  Emotion arc: {'✓' if emotion_arc else '✗'}")
    
    # Build prompt
    print("\nBuilding evaluation prompt...")
    prompt = build_judge_prompt(analysis, transcript, emotion_arc)
    
    # Run evaluation
    print(f"\nRunning evaluation with {model}...")
    if model == "gpt4":
        evaluation = evaluate_with_gpt4(prompt)
    else:
        evaluation = evaluate_with_claude(prompt)
    
    if "error" in evaluation:
        print(f"❌ Error: {evaluation['error']}")
        return evaluation
    
    # Calculate aggregate score
    aggregate = calculate_aggregate_score(evaluation)
    evaluation["aggregate_score"] = aggregate
    
    # Validate against human labels if requested
    if validate:
        print("\nValidating against human labels...")
        narrative_labels = load_json(LABELS_DIR / "narrative_labels.json")
        human_labels = narrative_labels.get("labels", {}).get("scene_analysis") if narrative_labels else None
        
        if human_labels:
            validation = validate_against_human(evaluation, human_labels)
            evaluation["human_validation"] = validation
            print(f"  Agreement rate: {validation.get('agreement_rate', 'N/A'):.0%}")
        else:
            print("  ⚠️ No human labels found for validation")
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    scores = evaluation.get("scores", {})
    print("\n📊 Dimension Scores:")
    for criterion, data in scores.items():
        score = data.get("score")
        if score is not None:
            bar = "█" * int(score) + "░" * (5 - int(score))
            print(f"   {criterion.replace('_', ' ').title():25} {score}/5 [{bar}]")
            print(f"      └─ {data.get('justification', 'N/A')[:60]}...")
    
    print(f"\n📈 Aggregate Score: {aggregate}/5")
    
    if evaluation.get("useful_to_reader"):
        print("\n✅ Analysis is useful to a reader")
    else:
        print("\n⚠️ Analysis may need improvement to be useful")
    
    print(f"\n💪 Strengths:")
    for s in evaluation.get("strengths", [])[:3]:
        print(f"   • {s}")
    
    print(f"\n📝 Areas for Improvement:")
    for w in evaluation.get("weaknesses", [])[:3]:
        print(f"   • {w}")
    
    # Save results
    evaluation["timestamp"] = datetime.now().isoformat()
    output = Path(output_path) if output_path else RESULTS_DIR / "llm_judge_evaluation.json"
    save_json(evaluation, output)
    print(f"\n💾 Results saved to {output}")
    
    return evaluation


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LLM-as-Judge Evaluation")
    parser.add_argument("--model", choices=["gpt4", "claude"], default="gpt4",
                       help="Which LLM to use as judge")
    parser.add_argument("--validate", action="store_true",
                       help="Validate against human labels")
    parser.add_argument("--output", help="Output path for results")
    args = parser.parse_args()
    
    run_llm_judge(model=args.model, validate=args.validate, output_path=args.output)


if __name__ == "__main__":
    main()
