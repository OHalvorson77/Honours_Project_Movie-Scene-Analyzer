import json
from typing import List, Dict, Any

# Helper function to load the paired_data.json file and data
def load_paired_data(path: str = "paired_data.json") -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def extract_keyframes(
    paired_data: List[Dict[str, Any]],
    max_frames_per_segment: int = 2,
    include_conflicts: bool = True,
    min_confidence: float = 0.5
) -> List[Dict[str, Any]]:

    keyframes = []
    prev_emotion = None
    
    for i, segment in enumerate(paired_data):
        frames = segment.get("frames", [])
        if not frames:
            continue
        
        # Determining the segment importance
        is_conflict = segment.get("conflict_type") is not None
        is_transition = prev_emotion and prev_emotion != segment.get("fused_emotion")
        confidence = segment.get("face_confidence", 0)
        
        # Calculating the priority score
        priority = 0
        if is_conflict:
            priority += 3
        if is_transition:
            priority += 2
        if confidence >= min_confidence:
            priority += 1
        
        # Select frames from this segment
        selected_frames = []
        
        if is_conflict and include_conflicts:
            # For conflicts, take first and middle frame to show the mismatch
            selected_frames.append(frames[0])
            if len(frames) > 2:
                selected_frames.append(frames[len(frames) // 2])
        elif priority > 0:
            # Take representative frames
            selected_frames.append(frames[0])  # First frame
            if len(frames) > 3 and max_frames_per_segment > 1:
                selected_frames.append(frames[len(frames) // 2])  # Middle frame
        else:
            # Low priority - just take one frame
            selected_frames.append(frames[len(frames) // 2])
        
        # Limit frames per segment
        selected_frames = selected_frames[:max_frames_per_segment]
        
        # Building the keyframe entries
        for frame in selected_frames:
            keyframes.append({
                "frame": frame,
                "segment_index": i,
                "text": segment.get("text", ""),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "speech_emotion": segment.get("speech_emotion"),
                "face_emotion": segment.get("face_emotion"),
                "fused_emotion": segment.get("fused_emotion"),
                "conflict_type": segment.get("conflict_type"),
                "is_transition": is_transition,
                "priority": priority
            })
        
        prev_emotion = segment.get("fused_emotion")
    
    return keyframes


def get_conflict_summary(paired_data: List[Dict[str, Any]]) -> Dict[str, Any]:

    # This function generates and returns a summary of the emotion conflicts from paired.json
    conflicts = []
    
    for i, segment in enumerate(paired_data):
        if segment.get("conflict_type"):
            conflicts.append({
                "segment": i,
                "time": f"{segment.get('start', 0):.1f}s - {segment.get('end', 0):.1f}s",
                "text": segment.get("text", ""),
                "conflict_type": segment.get("conflict_type"),
                "speech_emotion": segment.get("speech_emotion"),
                "face_emotion": segment.get("face_emotion"),
                "speech_confidence": segment.get("speech_confidence"),
                "face_confidence": segment.get("face_confidence")
            })
    
    return {
        "total_segments": len(paired_data),
        "conflict_count": len(conflicts),
        "conflicts": conflicts
    }


# This function saves the keyframes to json
def save_keyframes(keyframes: List[Dict[str, Any]], output_path: str = "keyframes.json"):
    
    with open(output_path, "w") as f:
        json.dump(keyframes, f, indent=2)
    print(f"Saved {len(keyframes)} keyframes to {output_path}")


def main():
    print("Loading paired data...")
    paired_data = load_paired_data()
    
    print(f"Processing {len(paired_data)} segments...")
    keyframes = extract_keyframes(paired_data)
    
    print(f"\nKeyframe extraction complete:")
    print(f"  Total segments: {len(paired_data)}")
    print(f"  Keyframes selected: {len(keyframes)}")
    
    # Count by priority
    conflict_frames = sum(1 for kf in keyframes if kf.get("conflict_type"))
    transition_frames = sum(1 for kf in keyframes if kf.get("is_transition"))
    print(f"  Conflict frames: {conflict_frames}")
    print(f"  Transition frames: {transition_frames}")
    
    # Generate conflict summary
    conflict_summary = get_conflict_summary(paired_data)
    print(f"\nConflict summary:")
    print(f"  {conflict_summary['conflict_count']} segments with audio/visual emotion mismatch")
    
    save_keyframes(keyframes)
    
    # Also save conflict summary for GPT-4o context
    with open("conflict_summary.json", "w") as f:
        json.dump(conflict_summary, f, indent=2)
    print(f"Saved conflict summary to conflict_summary.json")
    
    return keyframes


if __name__ == "__main__":
    main()

