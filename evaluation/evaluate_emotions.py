"""
Emotion Classification Evaluation

Compares model predictions against ground truth labels to calculate:
- Overall accuracy
- Per-class precision, recall, F1
- Confusion matrix
- Confidence calibration

Usage:
    python evaluate_emotions.py
    python evaluate_emotions.py --output results/emotion_eval.json
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

# Paths
LABELS_DIR = Path(__file__).parent / "labels"
PAIRED_DATA_PATH = Path(__file__).parent.parent / "paired_data.json"
RESULTS_DIR = Path(__file__).parent / "results"

EMOTIONS = ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def load_json(path: Path) -> Any:
    """Load JSON file."""
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_json(data: Any, path: Path):
    """Save data to JSON."""
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def calculate_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = EMOTIONS
) -> Dict[str, Any]:
    """
    Calculate classification metrics.
    
    Returns:
        Dict with accuracy, per-class metrics, and confusion matrix
    """
    # Overall accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0
    
    # Per-class metrics
    per_class = {}
    confusion = defaultdict(lambda: defaultdict(int))
    
    for emotion in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == emotion and p == emotion)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != emotion and p == emotion)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == emotion and p != emotion)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = sum(1 for t in y_true if t == emotion)
        
        per_class[emotion] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": support
        }
    
    # Confusion matrix
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1
    
    # Convert to regular dict
    confusion_matrix = {k: dict(v) for k, v in confusion.items()}
    
    # Macro averages
    macro_precision = sum(m["precision"] for m in per_class.values()) / len(per_class)
    macro_recall = sum(m["recall"] for m in per_class.values()) / len(per_class)
    macro_f1 = sum(m["f1"] for m in per_class.values()) / len(per_class)
    
    # Weighted averages
    total_support = sum(m["support"] for m in per_class.values())
    if total_support > 0:
        weighted_precision = sum(m["precision"] * m["support"] for m in per_class.values()) / total_support
        weighted_recall = sum(m["recall"] * m["support"] for m in per_class.values()) / total_support
        weighted_f1 = sum(m["f1"] * m["support"] for m in per_class.values()) / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0
    
    return {
        "accuracy": round(accuracy, 3),
        "total_samples": len(y_true),
        "correct": correct,
        "per_class": per_class,
        "macro_avg": {
            "precision": round(macro_precision, 3),
            "recall": round(macro_recall, 3),
            "f1": round(macro_f1, 3)
        },
        "weighted_avg": {
            "precision": round(weighted_precision, 3),
            "recall": round(weighted_recall, 3),
            "f1": round(weighted_f1, 3)
        },
        "confusion_matrix": confusion_matrix
    }


def evaluate_confidence_calibration(
    predictions: List[Dict[str, Any]],
    labels: Dict[str, str]
) -> Dict[str, Any]:
    """
    Evaluate how well confidence scores correlate with accuracy.
    
    Well-calibrated: high confidence predictions should be more accurate.
    """
    # Group by confidence bins
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    bin_results = []
    
    for low, high in bins:
        bin_preds = [
            p for p in predictions
            if low <= p.get("confidence", 0) < high
        ]
        
        if bin_preds:
            correct = sum(
                1 for p in bin_preds
                if p["predicted"] == labels.get(f"seg_{p['index']}")
            )
            bin_results.append({
                "confidence_range": f"{low:.1f}-{high:.1f}",
                "count": len(bin_preds),
                "accuracy": round(correct / len(bin_preds), 3),
                "avg_confidence": round(sum(p.get("confidence", 0) for p in bin_preds) / len(bin_preds), 3)
            })
    
    return {"calibration_bins": bin_results}


def evaluate_by_source(
    paired_data: List[Dict],
    labels: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    Compare accuracy of speech vs face emotion.
    """
    speech_correct = 0
    speech_total = 0
    face_correct = 0
    face_total = 0
    fused_correct = 0
    fused_total = 0
    
    for key, label_data in labels.items():
        idx = label_data.get("segment_index")
        if idx is None or idx >= len(paired_data):
            continue
        
        seg = paired_data[idx]
        true_label = label_data.get("label")
        
        if true_label:
            # Speech emotion
            if seg.get("speech_emotion"):
                speech_total += 1
                if seg["speech_emotion"] == true_label:
                    speech_correct += 1
            
            # Face emotion
            if seg.get("face_emotion"):
                face_total += 1
                if seg["face_emotion"] == true_label:
                    face_correct += 1
            
            # Fused emotion
            if seg.get("fused_emotion"):
                fused_total += 1
                if seg["fused_emotion"] == true_label:
                    fused_correct += 1
    
    return {
        "speech_emotion": {
            "accuracy": round(speech_correct / speech_total, 3) if speech_total > 0 else 0,
            "total": speech_total,
            "correct": speech_correct
        },
        "face_emotion": {
            "accuracy": round(face_correct / face_total, 3) if face_total > 0 else 0,
            "total": face_total,
            "correct": face_correct
        },
        "fused_emotion": {
            "accuracy": round(fused_correct / fused_total, 3) if fused_total > 0 else 0,
            "total": fused_total,
            "correct": fused_correct
        }
    }


def run_evaluation(output_path: str = None) -> Dict[str, Any]:
    """Run the full emotion evaluation."""
    print("=" * 60)
    print("EMOTION CLASSIFICATION EVALUATION")
    print("=" * 60)
    
    # Load data
    paired_data = load_json(PAIRED_DATA_PATH)
    emotion_labels = load_json(LABELS_DIR / "emotion_labels.json")
    
    if not paired_data:
        print("\n❌ Error: paired_data.json not found")
        return {"error": "No paired data"}
    
    if not emotion_labels or not emotion_labels.get("labels"):
        print("\n❌ Error: No emotion labels found")
        print("Run: python labeling_tool.py --mode emotion")
        return {"error": "No labels"}
    
    labels = emotion_labels["labels"]
    print(f"\nFound {len(labels)} labeled segments")
    
    # Prepare data
    y_true = []
    y_pred = []
    predictions = []
    
    for key, label_data in labels.items():
        idx = label_data.get("segment_index")
        true_label = label_data.get("label")
        
        if idx is None or true_label is None or true_label == "skip":
            continue
        
        if idx < len(paired_data):
            seg = paired_data[idx]
            pred_label = seg.get("fused_emotion") or seg.get("speech_emotion")
            
            if pred_label:
                y_true.append(true_label)
                y_pred.append(pred_label)
                predictions.append({
                    "index": idx,
                    "predicted": pred_label,
                    "confidence": seg.get("fused_confidence", 0)
                })
    
    print(f"Evaluating {len(y_true)} samples\n")
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Source comparison
    source_comparison = evaluate_by_source(paired_data, labels)
    
    # Confidence calibration
    calibration = evaluate_confidence_calibration(predictions, 
        {f"seg_{l['segment_index']}": l["label"] for l in labels.values() if l.get("label") != "skip"})
    
    # Combine results
    results = {
        "overall": metrics,
        "by_source": source_comparison,
        "confidence_calibration": calibration,
        "metadata": {
            "total_labeled": len(labels),
            "evaluated": len(y_true),
            "skipped": len(labels) - len(y_true)
        }
    }
    
    # Print results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Overall Accuracy: {metrics['accuracy']:.1%}")
    print(f"   ({metrics['correct']}/{metrics['total_samples']} correct)")
    
    print(f"\n📈 Macro F1: {metrics['macro_avg']['f1']:.3f}")
    print(f"   Weighted F1: {metrics['weighted_avg']['f1']:.3f}")
    
    print("\n📋 Per-Class Performance:")
    for emotion, m in sorted(metrics["per_class"].items(), key=lambda x: x[1]["f1"], reverse=True):
        if m["support"] > 0:
            print(f"   {emotion:10} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} (n={m['support']})")
    
    print("\n🔍 By Emotion Source:")
    for source, m in source_comparison.items():
        print(f"   {source:15} {m['accuracy']:.1%} ({m['correct']}/{m['total']})")
    
    # Save results
    if output_path:
        save_json(results, Path(output_path))
        print(f"\n💾 Results saved to {output_path}")
    else:
        default_path = RESULTS_DIR / "emotion_evaluation.json"
        save_json(results, default_path)
        print(f"\n💾 Results saved to {default_path}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate emotion classification")
    parser.add_argument("--output", help="Output path for results")
    args = parser.parse_args()
    
    run_evaluation(args.output)


if __name__ == "__main__":
    main()
