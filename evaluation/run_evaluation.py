"""
Evaluation Runner

Runs all evaluation metrics and generates a comprehensive report.

Usage:
    python run_evaluation.py                    # Run all evaluations
    python run_evaluation.py --report           # Generate PDF-style report
    python run_evaluation.py --llm-judge        # Include LLM-as-judge
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from evaluate_emotions import run_evaluation as eval_emotions
from evaluate_conflicts import run_evaluation as eval_conflicts
from evaluate_narrative import run_evaluation as eval_narrative
from labeling_tool import show_status, LABELS_DIR

RESULTS_DIR = Path(__file__).parent / "results"


def run_all_evaluations(use_llm_judge: bool = False) -> Dict[str, Any]:
    """Run all evaluation modules and combine results."""
    print("\n" + "=" * 70)
    print("   COMPREHENSIVE EVALUATION SUITE")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check labeling status first
    print("\n" + "-" * 70)
    show_status()
    print("-" * 70)
    
    input("\nPress Enter to continue with evaluation...")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "evaluations": {}
    }
    
    # 1. Emotion evaluation
    print("\n\n")
    try:
        emotion_results = eval_emotions()
        results["evaluations"]["emotion"] = emotion_results
    except Exception as e:
        print(f"Error in emotion evaluation: {e}")
        results["evaluations"]["emotion"] = {"error": str(e)}
    
    # 2. Conflict evaluation
    print("\n\n")
    try:
        conflict_results = eval_conflicts()
        results["evaluations"]["conflict"] = conflict_results
    except Exception as e:
        print(f"Error in conflict evaluation: {e}")
        results["evaluations"]["conflict"] = {"error": str(e)}
    
    # 3. Narrative evaluation
    print("\n\n")
    try:
        narrative_results = eval_narrative(use_llm_judge=use_llm_judge)
        results["evaluations"]["narrative"] = narrative_results
    except Exception as e:
        print(f"Error in narrative evaluation: {e}")
        results["evaluations"]["narrative"] = {"error": str(e)}
    
    # Generate summary
    results["summary"] = generate_summary(results["evaluations"])
    
    # Save combined results
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "full_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print final summary
    print_final_summary(results)
    
    return results


def generate_summary(evaluations: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary metrics from all evaluations."""
    summary = {
        "metrics": {},
        "scores": {},
        "recommendations": []
    }
    
    # Emotion metrics
    if "emotion" in evaluations and "error" not in evaluations["emotion"]:
        emotion = evaluations["emotion"]
        if "overall" in emotion:
            summary["metrics"]["emotion_accuracy"] = emotion["overall"].get("accuracy", 0)
            summary["metrics"]["emotion_f1"] = emotion["overall"].get("weighted_avg", {}).get("f1", 0)
            
            # Score (convert to 1-5 scale)
            acc = emotion["overall"].get("accuracy", 0)
            summary["scores"]["emotion"] = round(acc * 5, 1)
            
            if acc < 0.5:
                summary["recommendations"].append("Emotion accuracy is low. Consider fine-tuning the emotion classifier.")
    
    # Conflict metrics
    if "conflict" in evaluations and "error" not in evaluations["conflict"]:
        conflict = evaluations["conflict"]
        if "summary" in conflict:
            precision = conflict["summary"].get("precision", 0)
            summary["metrics"]["conflict_precision"] = precision
            summary["scores"]["conflict"] = round(precision * 5, 1)
            
            if precision < 0.6:
                summary["recommendations"].append("Conflict detection has many false positives. Review detection thresholds.")
    
    # Narrative metrics
    if "narrative" in evaluations and "error" not in evaluations["narrative"]:
        narrative = evaluations["narrative"]
        if "summary" in narrative:
            avg = narrative["summary"].get("average_score", 0)
            summary["metrics"]["narrative_quality"] = avg
            summary["scores"]["narrative"] = avg
            
            if avg < 3:
                summary["recommendations"].append("Narrative quality is below average. Review LLM prompts.")
    
    # Overall score
    scores = [v for v in summary["scores"].values() if v > 0]
    if scores:
        summary["overall_score"] = round(sum(scores) / len(scores), 2)
    
    return summary


def print_final_summary(results: Dict[str, Any]):
    """Print the final evaluation summary."""
    print("\n\n")
    print("=" * 70)
    print("   FINAL EVALUATION SUMMARY")
    print("=" * 70)
    
    summary = results.get("summary", {})
    scores = summary.get("scores", {})
    metrics = summary.get("metrics", {})
    
    print("\n📊 SCORES (out of 5):")
    print("-" * 40)
    
    if "emotion" in scores:
        bar = "█" * int(scores["emotion"]) + "░" * (5 - int(scores["emotion"]))
        print(f"   Emotion Classification: {scores['emotion']:.1f} [{bar}]")
    
    if "conflict" in scores:
        bar = "█" * int(scores["conflict"]) + "░" * (5 - int(scores["conflict"]))
        print(f"   Conflict Detection:     {scores['conflict']:.1f} [{bar}]")
    
    if "narrative" in scores:
        bar = "█" * int(scores["narrative"]) + "░" * (5 - int(scores["narrative"]))
        print(f"   Narrative Quality:      {scores['narrative']:.1f} [{bar}]")
    
    if "overall_score" in summary:
        print("-" * 40)
        overall = summary["overall_score"]
        bar = "█" * int(overall) + "░" * (5 - int(overall))
        print(f"   OVERALL:                {overall:.2f} [{bar}]")
    
    print("\n📈 KEY METRICS:")
    print("-" * 40)
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"   {name.replace('_', ' ').title()}: {value:.1%}")
    
    recommendations = summary.get("recommendations", [])
    if recommendations:
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 40)
        for rec in recommendations:
            print(f"   • {rec}")
    
    print("\n" + "=" * 70)
    print(f"   Results saved to: {RESULTS_DIR / 'full_evaluation.json'}")
    print("=" * 70)


def generate_report(results: Dict[str, Any] = None) -> str:
    """Generate a text report suitable for sharing."""
    if results is None:
        # Load from file
        results_path = RESULTS_DIR / "full_evaluation.json"
        if results_path.exists():
            with open(results_path, "r") as f:
                results = json.load(f)
        else:
            return "No evaluation results found. Run evaluation first."
    
    summary = results.get("summary", {})
    
    report = []
    report.append("=" * 60)
    report.append("MOVIE SCENE ANALYZER - EVALUATION REPORT")
    report.append("=" * 60)
    report.append(f"\nDate: {results.get('timestamp', 'N/A')}")
    
    report.append("\n\n## Summary Scores\n")
    for name, score in summary.get("scores", {}).items():
        report.append(f"- {name.title()}: {score:.1f}/5")
    
    if "overall_score" in summary:
        report.append(f"\n**Overall Score: {summary['overall_score']:.2f}/5**")
    
    report.append("\n\n## Key Metrics\n")
    for name, value in summary.get("metrics", {}).items():
        if isinstance(value, float):
            report.append(f"- {name.replace('_', ' ').title()}: {value:.1%}")
    
    # Detailed results
    evaluations = results.get("evaluations", {})
    
    if "emotion" in evaluations and "overall" in evaluations["emotion"]:
        report.append("\n\n## Emotion Classification\n")
        emotion = evaluations["emotion"]["overall"]
        report.append(f"- Accuracy: {emotion.get('accuracy', 0):.1%}")
        report.append(f"- F1 Score: {emotion.get('weighted_avg', {}).get('f1', 0):.3f}")
        report.append(f"- Samples evaluated: {emotion.get('total_samples', 0)}")
    
    if "conflict" in evaluations and "summary" in evaluations["conflict"]:
        report.append("\n\n## Conflict Detection\n")
        conflict = evaluations["conflict"]["summary"]
        report.append(f"- Precision: {conflict.get('precision', 0):.1%}")
        report.append(f"- Valid conflicts: {conflict.get('valid', 0)}")
        report.append(f"- Invalid conflicts: {conflict.get('invalid', 0)}")
    
    if "narrative" in evaluations and "summary" in evaluations["narrative"]:
        report.append("\n\n## Narrative Quality\n")
        narrative = evaluations["narrative"]["summary"]
        if "human_score" in narrative:
            report.append(f"- Human rating: {narrative['human_score']}/5")
        if "llm_score" in narrative:
            report.append(f"- LLM judge rating: {narrative['llm_score']}/5")
    
    recommendations = summary.get("recommendations", [])
    if recommendations:
        report.append("\n\n## Recommendations\n")
        for rec in recommendations:
            report.append(f"- {rec}")
    
    report.append("\n\n" + "=" * 60)
    
    report_text = "\n".join(report)
    
    # Save report
    report_path = RESULTS_DIR / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    
    print(f"Report saved to {report_path}")
    
    return report_text


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run comprehensive evaluation")
    parser.add_argument("--llm-judge", action="store_true",
                       help="Include LLM-as-judge evaluation")
    parser.add_argument("--report", action="store_true",
                       help="Generate text report from existing results")
    args = parser.parse_args()
    
    if args.report:
        report = generate_report()
        print(report)
    else:
        results = run_all_evaluations(use_llm_judge=args.llm_judge)
        
        print("\n\nGenerate text report? (y/n): ", end="")
        if input().strip().lower() == 'y':
            report = generate_report(results)
            print("\n" + report)


if __name__ == "__main__":
    main()
