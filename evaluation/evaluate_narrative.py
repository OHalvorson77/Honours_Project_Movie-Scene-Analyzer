"""
Narrative Quality Evaluation

Evaluates the quality of AI-generated scene analysis using:
1. Human ratings (from labeling tool)
2. LLM-as-judge (optional - uses Claude to rate GPT-4o output)
3. Cross-validation agreement metrics

Usage:
    python evaluate_narrative.py
    python evaluate_narrative.py --llm-judge  # Use LLM to evaluate
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

LABELS_DIR = Path(__file__).parent / "labels"
RESULTS_DIR = Path(__file__).parent / "results"
SCENE_ANALYSIS_PATH = Path(__file__).parent.parent / "scene_analysis.json"
CLAUDE_ANALYSIS_PATH = Path(__file__).parent.parent / "claude_analysis.json"
COMPARISON_PATH = Path(__file__).parent.parent / "analysis_comparison.json"
REFINED_NARRATIVE_PATH = Path(__file__).parent.parent / "refined_narrative.json"


def load_json(path: Path) -> Any:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_json(data: Any, path: Path):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_human_ratings() -> Optional[Dict[str, Any]]:
    """Get human ratings from labeling tool."""
    narrative_labels = load_json(LABELS_DIR / "narrative_labels.json")
    if narrative_labels and "labels" in narrative_labels:
        return narrative_labels["labels"].get("scene_analysis")
    return None


def get_cross_validation_metrics() -> Dict[str, Any]:
    """Extract metrics from cross-validation comparison."""
    comparison = load_json(COMPARISON_PATH)
    if not comparison:
        return {"available": False}
    
    stats = comparison.get("statistics", {})
    
    return {
        "available": True,
        "agreement_rate": stats.get("agreement_rate", 0),
        "agreements": stats.get("agreements", 0),
        "differences": stats.get("differences", 0),
        "themes_overlap": len(comparison.get("themes", {}).get("common", [])),
        "conflict_interpretations_count": len(comparison.get("conflict_interpretations", []))
    }


def llm_judge_evaluation(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use an LLM to evaluate the narrative quality.
    This provides an automated alternative to human evaluation.
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        return {"error": "anthropic package not installed"}
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not set"}
    
    client = Anthropic(api_key=api_key)
    
    # Build the analysis text
    analysis_text = json.dumps(analysis, indent=2)[:4000]  # Limit size
    
    prompt = f"""You are evaluating the quality of an AI-generated movie scene analysis.

ANALYSIS TO EVALUATE:
{analysis_text}

Rate each aspect from 1-5:
1 = Poor
2 = Below Average  
3 = Average
4 = Good
5 = Excellent

Provide your ratings in this exact JSON format:
{{
    "coherence": <1-5>,
    "coherence_explanation": "<brief explanation>",
    "accuracy": <1-5>,
    "accuracy_explanation": "<brief explanation>",
    "insight": <1-5>,
    "insight_explanation": "<brief explanation>",
    "emotion_interpretation": <1-5>,
    "emotion_interpretation_explanation": "<brief explanation>",
    "overall": <1-5>,
    "overall_explanation": "<brief explanation>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"]
}}

Respond ONLY with the JSON, no other text."""

    print("Calling Claude for LLM-as-judge evaluation...")
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    try:
        result = json.loads(response.content[0].text)
        result["judge_model"] = "claude-3-5-sonnet"
        return result
    except json.JSONDecodeError:
        return {"error": "Failed to parse LLM response", "raw": response.content[0].text}


def calculate_overall_score(ratings: Dict[str, int]) -> float:
    """Calculate weighted overall score from component ratings."""
    weights = {
        "coherence": 0.25,
        "accuracy": 0.25,
        "insight": 0.20,
        "emotion_accuracy": 0.15,
        "conflict_interpretation": 0.15
    }
    
    total = 0
    weight_sum = 0
    for key, weight in weights.items():
        if key in ratings:
            total += ratings[key] * weight
            weight_sum += weight
    
    return round(total / weight_sum, 2) if weight_sum > 0 else 0


def run_evaluation(use_llm_judge: bool = False, output_path: str = None) -> Dict[str, Any]:
    """Run narrative quality evaluation."""
    print("=" * 60)
    print("NARRATIVE QUALITY EVALUATION")
    print("=" * 60)
    
    results = {
        "human_evaluation": None,
        "llm_evaluation": None,
        "cross_validation": None,
        "summary": {}
    }
    
    # 1. Human ratings
    print("\n📋 Checking human ratings...")
    human_ratings = get_human_ratings()
    
    if human_ratings:
        results["human_evaluation"] = human_ratings
        print(f"   Overall score: {human_ratings.get('overall_score', 'N/A')}/5")
        for criterion, score in human_ratings.get("ratings", {}).items():
            print(f"   {criterion}: {score}/5")
    else:
        print("   ⚠️  No human ratings found")
        print("   Run: python labeling_tool.py --mode narrative")
    
    # 2. Cross-validation metrics
    print("\n🔄 Checking cross-validation...")
    cv_metrics = get_cross_validation_metrics()
    results["cross_validation"] = cv_metrics
    
    if cv_metrics.get("available"):
        print(f"   Agreement rate: {cv_metrics['agreement_rate']:.0%}")
        print(f"   Shared themes: {cv_metrics['themes_overlap']}")
    else:
        print("   ⚠️  No cross-validation data found")
    
    # 3. LLM-as-judge (improved rubric-based evaluation)
    if use_llm_judge:
        print("\n🤖 Running LLM-as-judge evaluation (rubric-based)...")
        try:
            from llm_judge import run_llm_judge
            llm_result = run_llm_judge(model="gpt4", validate=human_ratings is not None)
            results["llm_evaluation"] = llm_result
            
            if "error" not in llm_result:
                print(f"   Aggregate score: {llm_result.get('aggregate_score', 'N/A')}/5")
                if "human_validation" in llm_result:
                    print(f"   Human correlation: {llm_result['human_validation'].get('agreement_rate', 0):.0%}")
        except ImportError:
            # Fallback to simple evaluation
            scene_analysis = load_json(SCENE_ANALYSIS_PATH)
            if scene_analysis:
                llm_result = llm_judge_evaluation(scene_analysis)
                results["llm_evaluation"] = llm_result
                if "error" not in llm_result:
                    print(f"   Overall: {llm_result.get('overall', 'N/A')}/5")
            else:
                print("   ⚠️  No scene analysis found")
    
    # 4. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    scores = []
    
    if human_ratings:
        human_score = human_ratings.get("overall_score", 0)
        scores.append(("Human Rating", human_score))
        results["summary"]["human_score"] = human_score
    
    if results.get("llm_evaluation") and "overall" in results["llm_evaluation"]:
        llm_score = results["llm_evaluation"]["overall"]
        scores.append(("LLM Judge", llm_score))
        results["summary"]["llm_score"] = llm_score
    
    if cv_metrics.get("available"):
        # Convert agreement rate to 1-5 scale
        cv_score = round(cv_metrics["agreement_rate"] * 5, 1)
        scores.append(("Cross-validation", cv_score))
        results["summary"]["cv_score"] = cv_score
    
    if scores:
        avg_score = sum(s[1] for s in scores) / len(scores)
        results["summary"]["average_score"] = round(avg_score, 2)
        
        print(f"\n📊 Scores:")
        for name, score in scores:
            print(f"   {name}: {score}/5")
        print(f"\n   Average: {avg_score:.2f}/5")
        
        # Interpretation
        print("\n🎯 Interpretation:")
        if avg_score >= 4:
            print("   ✅ Excellent narrative quality!")
        elif avg_score >= 3:
            print("   ⚠️  Good quality, room for improvement")
        else:
            print("   ❌ Below average, needs significant improvement")
    else:
        print("\n⚠️  No evaluation data available")
        print("   Run labeling tool or enable --llm-judge")
    
    # Save results
    output = Path(output_path) if output_path else RESULTS_DIR / "narrative_evaluation.json"
    save_json(results, output)
    print(f"\n💾 Results saved to {output}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate narrative quality")
    parser.add_argument("--llm-judge", action="store_true", 
                       help="Use LLM (Claude) to evaluate narrative quality")
    parser.add_argument("--output", help="Output path for results")
    args = parser.parse_args()
    
    run_evaluation(use_llm_judge=args.llm_judge, output_path=args.output)


if __name__ == "__main__":
    main()
