"""
Ground Truth Labeling Tool

Interactive CLI tool to create ground truth labels for evaluation.
Labels:
1. Emotion labels for each transcript segment
2. Conflict validity (is the detected conflict actually sarcasm/irony?)
3. Narrative quality ratings

Usage:
    python labeling_tool.py --mode emotion      # Label emotions
    python labeling_tool.py --mode conflict     # Validate conflicts
    python labeling_tool.py --mode narrative    # Rate narrative quality
    python labeling_tool.py --export            # Export all labels
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Emotion options
EMOTIONS = ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise", "skip"]

# Paths
LABELS_DIR = Path(__file__).parent / "labels"
PAIRED_DATA_PATH = Path(__file__).parent.parent / "paired_data.json"
SCENE_ANALYSIS_PATH = Path(__file__).parent.parent / "scene_analysis.json"
CONFLICT_SUMMARY_PATH = Path(__file__).parent.parent / "conflict_summary.json"


def ensure_labels_dir():
    """Create labels directory if it doesn't exist."""
    LABELS_DIR.mkdir(exist_ok=True)


def load_json(path: Path) -> Optional[Any]:
    """Load JSON file if it exists."""
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_json(data: Any, path: Path):
    """Save data to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_labels(label_type: str) -> Dict[str, Any]:
    """Load existing labels or create new structure."""
    ensure_labels_dir()
    path = LABELS_DIR / f"{label_type}_labels.json"
    
    if path.exists():
        return load_json(path)
    
    return {
        "type": label_type,
        "created_at": datetime.now().isoformat(),
        "labels": {},
        "metadata": {}
    }


def save_labels(labels: Dict[str, Any], label_type: str):
    """Save labels to file."""
    ensure_labels_dir()
    path = LABELS_DIR / f"{label_type}_labels.json"
    labels["updated_at"] = datetime.now().isoformat()
    save_json(labels, path)
    print(f"Saved {len(labels['labels'])} labels to {path}")


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


# ============================================================================
# Emotion Labeling
# ============================================================================

def label_emotions():
    """Interactive emotion labeling for transcript segments."""
    print("\n" + "=" * 60)
    print("EMOTION LABELING TOOL")
    print("=" * 60)
    print("\nYou will label the TRUE emotion for each transcript segment.")
    print("This creates ground truth data for evaluation.\n")
    
    # Load data
    paired_data = load_json(PAIRED_DATA_PATH)
    if not paired_data:
        print("Error: paired_data.json not found. Run the pipeline first.")
        return
    
    labels = load_labels("emotion")
    
    print(f"Total segments: {len(paired_data)}")
    print(f"Already labeled: {len(labels['labels'])}")
    print("\nOptions: " + ", ".join(f"{i}={e}" for i, e in enumerate(EMOTIONS)))
    print("Press 'q' to quit, 's' to skip, 'b' to go back\n")
    
    # Find first unlabeled segment
    start_idx = 0
    for i, seg in enumerate(paired_data):
        key = f"seg_{i}"
        if key not in labels["labels"]:
            start_idx = i
            break
    
    idx = start_idx
    while idx < len(paired_data):
        seg = paired_data[idx]
        key = f"seg_{idx}"
        
        # Check if already labeled
        existing = labels["labels"].get(key, {})
        existing_label = existing.get("label", "unlabeled")
        
        clear_screen()
        print(f"\n--- Segment {idx + 1}/{len(paired_data)} ---")
        print(f"Time: {seg['start']:.1f}s - {seg['end']:.1f}s")
        print(f"\nText: \"{seg['text']}\"")
        print(f"\nModel predictions:")
        print(f"  Speech emotion: {seg.get('speech_emotion', 'N/A')} (conf: {seg.get('speech_confidence', 0):.2f})")
        print(f"  Face emotion:   {seg.get('face_emotion', 'N/A')} (conf: {seg.get('face_confidence', 0):.2f})")
        print(f"  Fused emotion:  {seg.get('fused_emotion', 'N/A')}")
        if seg.get('conflict_type'):
            print(f"  ⚠️  Conflict: {seg['conflict_type']}")
        print(f"\nCurrent label: {existing_label}")
        print("\n" + "-" * 40)
        print("Options: " + ", ".join(f"{i}={e}" for i, e in enumerate(EMOTIONS)))
        
        user_input = input("\nYour label (0-8, q=quit, b=back): ").strip().lower()
        
        if user_input == 'q':
            break
        elif user_input == 'b':
            idx = max(0, idx - 1)
            continue
        elif user_input == 's' or user_input == '8':
            idx += 1
            continue
        elif user_input.isdigit() and 0 <= int(user_input) < len(EMOTIONS) - 1:
            emotion = EMOTIONS[int(user_input)]
            labels["labels"][key] = {
                "segment_index": idx,
                "text": seg["text"][:50],
                "label": emotion,
                "model_prediction": seg.get("fused_emotion"),
                "labeled_at": datetime.now().isoformat()
            }
            print(f"✓ Labeled as: {emotion}")
            idx += 1
        else:
            print("Invalid input. Try again.")
            continue
        
        # Auto-save every 10 labels
        if len(labels["labels"]) % 10 == 0:
            save_labels(labels, "emotion")
    
    save_labels(labels, "emotion")
    print(f"\nLabeling complete! {len(labels['labels'])} segments labeled.")


# ============================================================================
# Conflict Validation
# ============================================================================

def validate_conflicts():
    """Validate detected audio/visual conflicts."""
    print("\n" + "=" * 60)
    print("CONFLICT VALIDATION TOOL")
    print("=" * 60)
    print("\nYou will validate whether detected conflicts are genuine.")
    print("A conflict is valid if it represents sarcasm, irony, or suppressed emotion.\n")
    
    # Load data
    paired_data = load_json(PAIRED_DATA_PATH)
    if not paired_data:
        print("Error: paired_data.json not found.")
        return
    
    # Filter to only conflict segments
    conflicts = [(i, seg) for i, seg in enumerate(paired_data) if seg.get("conflict_type")]
    
    if not conflicts:
        print("No conflicts detected in the data.")
        return
    
    labels = load_labels("conflict")
    
    print(f"Total conflicts: {len(conflicts)}")
    print(f"Already validated: {len(labels['labels'])}")
    print("\nOptions:")
    print("  1 = Valid conflict (genuine sarcasm/irony/suppression)")
    print("  0 = Invalid (model error, not meaningful)")
    print("  ? = Uncertain")
    print("  q = Quit, b = Back\n")
    
    idx = 0
    while idx < len(conflicts):
        seg_idx, seg = conflicts[idx]
        key = f"conflict_{seg_idx}"
        
        existing = labels["labels"].get(key, {})
        existing_label = existing.get("valid", "unlabeled")
        
        clear_screen()
        print(f"\n--- Conflict {idx + 1}/{len(conflicts)} ---")
        print(f"Time: {seg['start']:.1f}s - {seg['end']:.1f}s")
        print(f"\nText: \"{seg['text']}\"")
        print(f"\nConflict details:")
        print(f"  Type: {seg['conflict_type']}")
        print(f"  Speech emotion: {seg.get('speech_emotion')} (conf: {seg.get('speech_confidence', 0):.2f})")
        print(f"  Face emotion:   {seg.get('face_emotion')} (conf: {seg.get('face_confidence', 0):.2f})")
        print(f"\nCurrent validation: {existing_label}")
        print("\n" + "-" * 40)
        print("Is this a valid/meaningful conflict?")
        
        user_input = input("\n1=Valid, 0=Invalid, ?=Uncertain, q=quit, b=back: ").strip().lower()
        
        if user_input == 'q':
            break
        elif user_input == 'b':
            idx = max(0, idx - 1)
            continue
        elif user_input in ['1', '0', '?']:
            valid_map = {'1': True, '0': False, '?': 'uncertain'}
            labels["labels"][key] = {
                "segment_index": seg_idx,
                "text": seg["text"][:50],
                "conflict_type": seg["conflict_type"],
                "valid": valid_map[user_input],
                "speech_emotion": seg.get("speech_emotion"),
                "face_emotion": seg.get("face_emotion"),
                "validated_at": datetime.now().isoformat()
            }
            idx += 1
        else:
            print("Invalid input.")
            continue
        
        if len(labels["labels"]) % 5 == 0:
            save_labels(labels, "conflict")
    
    save_labels(labels, "conflict")
    
    # Show summary
    valid_count = sum(1 for v in labels["labels"].values() if v.get("valid") == True)
    invalid_count = sum(1 for v in labels["labels"].values() if v.get("valid") == False)
    uncertain_count = sum(1 for v in labels["labels"].values() if v.get("valid") == "uncertain")
    
    print(f"\nValidation complete!")
    print(f"  Valid conflicts: {valid_count}")
    print(f"  Invalid conflicts: {invalid_count}")
    print(f"  Uncertain: {uncertain_count}")


# ============================================================================
# Narrative Quality Rating
# ============================================================================

def rate_narrative():
    """Rate the quality of generated narrative analysis."""
    print("\n" + "=" * 60)
    print("NARRATIVE QUALITY RATING TOOL")
    print("=" * 60)
    print("\nYou will rate the quality of the AI-generated scene analysis.\n")
    
    # Load scene analysis
    scene_analysis = load_json(SCENE_ANALYSIS_PATH)
    if not scene_analysis:
        print("Error: scene_analysis.json not found.")
        return
    
    labels = load_labels("narrative")
    
    # Define rating criteria
    criteria = [
        ("coherence", "Narrative Coherence", "Does the analysis flow logically and make sense?"),
        ("accuracy", "Factual Accuracy", "Does it accurately describe what happens in the scene?"),
        ("insight", "Depth of Insight", "Does it provide meaningful observations beyond the obvious?"),
        ("emotion_accuracy", "Emotion Accuracy", "Are the emotion interpretations reasonable?"),
        ("conflict_interpretation", "Conflict Interpretation", "Are audio/visual conflicts explained well?"),
    ]
    
    print("You will rate each aspect from 1-5:")
    print("  1 = Poor")
    print("  2 = Below Average")
    print("  3 = Average")
    print("  4 = Good")
    print("  5 = Excellent")
    print("\nPress Enter to continue...")
    input()
    
    clear_screen()
    
    # Show the analysis
    print("=" * 60)
    print("SCENE ANALYSIS TO RATE")
    print("=" * 60)
    
    if "summary" in scene_analysis:
        print(f"\nSummary:\n{scene_analysis['summary']}")
    
    if "scene_overview" in scene_analysis:
        overview = scene_analysis["scene_overview"]
        print(f"\nScene Overview:")
        print(f"  Setting: {overview.get('setting', 'N/A')}")
        print(f"  Mood: {overview.get('mood', 'N/A')}")
        print(f"  Atmosphere: {overview.get('atmosphere', 'N/A')}")
    
    if "themes" in scene_analysis:
        themes = scene_analysis["themes"]
        print(f"\nThemes: {', '.join(themes.get('central_themes', []))}")
    
    if "conflict_interpretations" in scene_analysis:
        print(f"\nConflict Interpretations: {len(scene_analysis['conflict_interpretations'])} provided")
    
    print("\n" + "=" * 60)
    print("\nNow rate each aspect (1-5):\n")
    
    ratings = {}
    for criterion_id, criterion_name, description in criteria:
        print(f"\n{criterion_name}")
        print(f"  {description}")
        
        while True:
            rating = input(f"  Rating (1-5): ").strip()
            if rating.isdigit() and 1 <= int(rating) <= 5:
                ratings[criterion_id] = int(rating)
                break
            print("  Please enter a number 1-5")
    
    # Optional comments
    print("\nAny additional comments? (press Enter to skip)")
    comments = input("> ").strip()
    
    # Calculate overall score
    overall = sum(ratings.values()) / len(ratings)
    
    labels["labels"]["scene_analysis"] = {
        "ratings": ratings,
        "overall_score": round(overall, 2),
        "comments": comments,
        "rated_at": datetime.now().isoformat()
    }
    
    save_labels(labels, "narrative")
    
    print(f"\n✓ Ratings saved!")
    print(f"  Overall score: {overall:.2f}/5")


# ============================================================================
# Export Labels
# ============================================================================

def export_labels():
    """Export all labels to a single JSON file for evaluation."""
    ensure_labels_dir()
    
    export_data = {
        "exported_at": datetime.now().isoformat(),
        "emotion_labels": load_labels("emotion"),
        "conflict_labels": load_labels("conflict"),
        "narrative_labels": load_labels("narrative")
    }
    
    # Summary stats
    emotion_count = len(export_data["emotion_labels"].get("labels", {}))
    conflict_count = len(export_data["conflict_labels"].get("labels", {}))
    narrative_rated = "scene_analysis" in export_data["narrative_labels"].get("labels", {})
    
    export_path = LABELS_DIR / "all_labels.json"
    save_json(export_data, export_path)
    
    print("\n" + "=" * 60)
    print("LABELS EXPORT SUMMARY")
    print("=" * 60)
    print(f"\nEmotion labels: {emotion_count}")
    print(f"Conflict validations: {conflict_count}")
    print(f"Narrative rated: {'Yes' if narrative_rated else 'No'}")
    print(f"\nExported to: {export_path}")


def show_status():
    """Show labeling progress status."""
    ensure_labels_dir()
    
    paired_data = load_json(PAIRED_DATA_PATH)
    emotion_labels = load_labels("emotion")
    conflict_labels = load_labels("conflict")
    narrative_labels = load_labels("narrative")
    
    total_segments = len(paired_data) if paired_data else 0
    total_conflicts = len([s for s in (paired_data or []) if s.get("conflict_type")])
    
    emotion_count = len(emotion_labels.get("labels", {}))
    conflict_count = len(conflict_labels.get("labels", {}))
    narrative_rated = "scene_analysis" in narrative_labels.get("labels", {})
    
    print("\n" + "=" * 60)
    print("LABELING STATUS")
    print("=" * 60)
    print(f"\nEmotion Labels:    {emotion_count}/{total_segments} segments")
    print(f"Conflict Validation: {conflict_count}/{total_conflicts} conflicts")
    print(f"Narrative Rating:   {'Complete' if narrative_rated else 'Not done'}")
    
    # Recommendation
    print("\n" + "-" * 60)
    print("RECOMMENDATION:")
    if emotion_count < 30:
        print(f"  → Label at least 30 emotion segments for reliable evaluation")
    if total_conflicts > 0 and conflict_count < total_conflicts:
        print(f"  → Validate all {total_conflicts} conflicts")
    if not narrative_rated:
        print(f"  → Rate the narrative quality")
    if emotion_count >= 30 and conflict_count >= total_conflicts and narrative_rated:
        print(f"  ✓ Ready to run evaluation!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Ground Truth Labeling Tool")
    parser.add_argument("--mode", choices=["emotion", "conflict", "narrative"],
                       help="Labeling mode")
    parser.add_argument("--export", action="store_true", help="Export all labels")
    parser.add_argument("--status", action="store_true", help="Show labeling status")
    
    args = parser.parse_args()
    
    if args.status:
        show_status()
    elif args.export:
        export_labels()
    elif args.mode == "emotion":
        label_emotions()
    elif args.mode == "conflict":
        validate_conflicts()
    elif args.mode == "narrative":
        rate_narrative()
    else:
        # Interactive menu
        print("\n" + "=" * 60)
        print("GROUND TRUTH LABELING TOOL")
        print("=" * 60)
        show_status()
        print("\n" + "-" * 60)
        print("OPTIONS:")
        print("  1. Label emotions")
        print("  2. Validate conflicts")
        print("  3. Rate narrative quality")
        print("  4. Export labels")
        print("  5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            label_emotions()
        elif choice == "2":
            validate_conflicts()
        elif choice == "3":
            rate_narrative()
        elif choice == "4":
            export_labels()
        else:
            print("Goodbye!")


if __name__ == "__main__":
    main()
