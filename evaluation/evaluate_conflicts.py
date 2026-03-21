"""
Conflict Detection Evaluation

Evaluates the multimodal conflict detection system:
- Precision: What % of detected conflicts are valid?
- Analysis of conflict types
- Comparison with ground truth validations

Usage:
    python evaluate_conflicts.py
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter

LABELS_DIR = Path(__file__).parent / "labels"
PAIRED_DATA_PATH = Path(__file__).parent.parent / "paired_data.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_json(path: Path) -> Any:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_json(data: Any, path: Path):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_evaluation(output_path: str = None) -> Dict[str, Any]:
    """Evaluate conflict detection performance."""
    print("=" * 60)
    print("CONFLICT DETECTION EVALUATION")
    print("=" * 60)
    
    # Load data
    paired_data = load_json(PAIRED_DATA_PATH)
    conflict_labels = load_json(LABELS_DIR / "conflict_labels.json")
    
    if not paired_data:
        print("\n❌ Error: paired_data.json not found")
        return {"error": "No paired data"}
    
    # Get all detected conflicts
    detected_conflicts = [
        (i, seg) for i, seg in enumerate(paired_data)
        if seg.get("conflict_type")
    ]
    
    print(f"\nTotal detected conflicts: {len(detected_conflicts)}")
    
    if not conflict_labels or not conflict_labels.get("labels"):
        print("\n⚠️  No conflict validations found")
        print("Run: python labeling_tool.py --mode conflict")
        
        # Still provide analysis of detected conflicts
        conflict_types = Counter(seg["conflict_type"] for _, seg in detected_conflicts)
        
        return {
            "detected_conflicts": len(detected_conflicts),
            "validated": 0,
            "conflict_type_distribution": dict(conflict_types),
            "note": "No ground truth labels - run labeling tool"
        }
    
    labels = conflict_labels["labels"]
    print(f"Validated conflicts: {len(labels)}")
    
    # Calculate metrics
    valid_count = sum(1 for v in labels.values() if v.get("valid") == True)
    invalid_count = sum(1 for v in labels.values() if v.get("valid") == False)
    uncertain_count = sum(1 for v in labels.values() if v.get("valid") == "uncertain")
    
    # Precision (excluding uncertain)
    evaluated = valid_count + invalid_count
    precision = valid_count / evaluated if evaluated > 0 else 0
    
    # Analysis by conflict type
    by_type = {}
    for key, label_data in labels.items():
        ctype = label_data.get("conflict_type", "unknown")
        if ctype not in by_type:
            by_type[ctype] = {"valid": 0, "invalid": 0, "uncertain": 0}
        
        validity = label_data.get("valid")
        if validity == True:
            by_type[ctype]["valid"] += 1
        elif validity == False:
            by_type[ctype]["invalid"] += 1
        else:
            by_type[ctype]["uncertain"] += 1
    
    # Calculate precision per type
    for ctype, counts in by_type.items():
        total = counts["valid"] + counts["invalid"]
        counts["precision"] = round(counts["valid"] / total, 3) if total > 0 else 0
        counts["total"] = total + counts["uncertain"]
    
    # Confidence analysis
    confidence_analysis = analyze_confidence_patterns(paired_data, labels)
    
    results = {
        "summary": {
            "detected_conflicts": len(detected_conflicts),
            "validated": len(labels),
            "valid": valid_count,
            "invalid": invalid_count,
            "uncertain": uncertain_count,
            "precision": round(precision, 3)
        },
        "by_conflict_type": by_type,
        "confidence_analysis": confidence_analysis
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Conflict Detection Precision: {precision:.1%}")
    print(f"   Valid: {valid_count}")
    print(f"   Invalid: {invalid_count}")
    print(f"   Uncertain: {uncertain_count}")
    
    print("\n📋 By Conflict Type:")
    for ctype, counts in sorted(by_type.items(), key=lambda x: x[1]["total"], reverse=True):
        print(f"   {ctype:20} Precision={counts['precision']:.0%} "
              f"({counts['valid']}/{counts['valid']+counts['invalid']} valid)")
    
    print("\n🔍 Interpretation:")
    if precision >= 0.8:
        print("   ✅ Excellent! Most detected conflicts are meaningful.")
    elif precision >= 0.6:
        print("   ⚠️  Good, but some false positives. Consider adjusting thresholds.")
    else:
        print("   ❌ Many false positives. Review conflict detection logic.")
    
    # Save results
    output = Path(output_path) if output_path else RESULTS_DIR / "conflict_evaluation.json"
    save_json(results, output)
    print(f"\n💾 Results saved to {output}")
    
    return results


def analyze_confidence_patterns(
    paired_data: List[Dict],
    labels: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    Analyze if confidence divergence correlates with valid conflicts.
    
    Hypothesis: Larger divergence between speech and face confidence
    might indicate more meaningful conflicts.
    """
    valid_divergences = []
    invalid_divergences = []
    
    for key, label_data in labels.items():
        idx = label_data.get("segment_index")
        if idx is None or idx >= len(paired_data):
            continue
        
        seg = paired_data[idx]
        speech_conf = seg.get("speech_confidence", 0) or 0
        face_conf = seg.get("face_confidence", 0) or 0
        divergence = abs(speech_conf - face_conf)
        
        if label_data.get("valid") == True:
            valid_divergences.append(divergence)
        elif label_data.get("valid") == False:
            invalid_divergences.append(divergence)
    
    avg_valid = sum(valid_divergences) / len(valid_divergences) if valid_divergences else 0
    avg_invalid = sum(invalid_divergences) / len(invalid_divergences) if invalid_divergences else 0
    
    return {
        "avg_confidence_divergence_valid": round(avg_valid, 3),
        "avg_confidence_divergence_invalid": round(avg_invalid, 3),
        "insight": "Higher divergence in valid conflicts" if avg_valid > avg_invalid else "No clear pattern"
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate conflict detection")
    parser.add_argument("--output", help="Output path for results")
    args = parser.parse_args()
    
    run_evaluation(args.output)


if __name__ == "__main__":
    main()
