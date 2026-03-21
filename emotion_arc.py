"""
Emotion Arc Extractor

Analyzes the emotional trajectory across the entire clip to:
1. Identify emotional phases (rising tension, climax, resolution)
2. Detect key emotional beats/turning points
3. Track per-character emotion arcs (when multiple characters exist)
4. Generate arc summary for narrative grounding

This addresses the professor's feedback on extracting the "emotional arc intact"
and maintaining narrative consistency.
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


# Emotion valence mapping (positive/negative/neutral)
EMOTION_VALENCE = {
    "happy": 1.0,
    "surprise": 0.5,
    "calm": 0.0,
    "neutral": 0.0,
    "sad": -0.5,
    "fear": -0.7,
    "angry": -0.8,
    "disgust": -0.6
}

# Emotion intensity (arousal level)
EMOTION_INTENSITY = {
    "angry": 0.9,
    "fear": 0.8,
    "surprise": 0.7,
    "happy": 0.6,
    "disgust": 0.5,
    "sad": 0.4,
    "calm": 0.1,
    "neutral": 0.2
}


def load_paired_data(path: str = "paired_data.json") -> List[Dict[str, Any]]:
    """Load the paired emotion data."""
    with open(path, "r") as f:
        return json.load(f)


def calculate_emotion_metrics(segment: Dict[str, Any]) -> Dict[str, float]:
    """Calculate valence and intensity for a segment."""
    emotion = segment.get("fused_emotion") or segment.get("speech_emotion") or "neutral"
    confidence = segment.get("fused_confidence", 0) or 0
    
    valence = EMOTION_VALENCE.get(emotion, 0)
    intensity = EMOTION_INTENSITY.get(emotion, 0)
    
    # Weight by confidence
    weighted_valence = valence * confidence
    weighted_intensity = intensity * confidence
    
    return {
        "valence": valence,
        "intensity": intensity,
        "weighted_valence": weighted_valence,
        "weighted_intensity": weighted_intensity,
        "emotion": emotion,
        "confidence": confidence
    }


def detect_turning_points(
    segments: List[Dict[str, Any]],
    valence_threshold: float = 0.3,
    intensity_threshold: float = 0.2
) -> List[Dict[str, Any]]:
    """
    Detect emotional turning points where significant shifts occur.
    
    A turning point is detected when:
    - Valence changes significantly (e.g., positive to negative)
    - Intensity spikes notably
    - Conflict type appears/disappears
    """
    turning_points = []
    
    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]
        
        prev_metrics = calculate_emotion_metrics(prev)
        curr_metrics = calculate_emotion_metrics(curr)
        
        valence_change = curr_metrics["valence"] - prev_metrics["valence"]
        intensity_change = curr_metrics["intensity"] - prev_metrics["intensity"]
        
        is_turning_point = False
        reasons = []
        
        # Significant valence shift
        if abs(valence_change) >= valence_threshold:
            is_turning_point = True
            direction = "positive" if valence_change > 0 else "negative"
            reasons.append(f"valence_shift_{direction}")
        
        # Intensity spike
        if intensity_change >= intensity_threshold:
            is_turning_point = True
            reasons.append("intensity_spike")
        
        # Conflict emergence
        if curr.get("conflict_type") and not prev.get("conflict_type"):
            is_turning_point = True
            reasons.append(f"conflict_start:{curr['conflict_type']}")
        
        # Conflict resolution
        if prev.get("conflict_type") and not curr.get("conflict_type"):
            is_turning_point = True
            reasons.append("conflict_resolved")
        
        if is_turning_point:
            turning_points.append({
                "segment_index": i,
                "timestamp": curr.get("start", 0),
                "text": curr.get("text", ""),
                "from_emotion": prev_metrics["emotion"],
                "to_emotion": curr_metrics["emotion"],
                "valence_change": round(valence_change, 3),
                "intensity_change": round(intensity_change, 3),
                "reasons": reasons,
                "significance": round(abs(valence_change) + abs(intensity_change), 3)
            })
    
    return sorted(turning_points, key=lambda x: x["significance"], reverse=True)


def identify_arc_phases(
    segments: List[Dict[str, Any]],
    window_size: int = 3
) -> List[Dict[str, Any]]:
    """
    Identify narrative arc phases using smoothed emotion trajectory.
    
    Phases:
    - exposition: Initial emotional baseline
    - rising_action: Increasing tension/intensity
    - climax: Peak emotional intensity
    - falling_action: Decreasing tension
    - resolution: Return to baseline or new equilibrium
    """
    if len(segments) < 5:
        return [{"phase": "short_clip", "segments": list(range(len(segments)))}]
    
    # Calculate smoothed intensity curve
    intensities = []
    for i, seg in enumerate(segments):
        metrics = calculate_emotion_metrics(seg)
        intensities.append(metrics["weighted_intensity"])
    
    # Smooth with moving average
    smoothed = []
    for i in range(len(intensities)):
        start = max(0, i - window_size // 2)
        end = min(len(intensities), i + window_size // 2 + 1)
        smoothed.append(sum(intensities[start:end]) / (end - start))
    
    # Find climax (peak intensity)
    climax_idx = smoothed.index(max(smoothed))
    
    # Define phase boundaries
    total = len(segments)
    phases = []
    
    # Exposition: first 15-20%
    expo_end = min(climax_idx, total // 5)
    if expo_end > 0:
        phases.append({
            "phase": "exposition",
            "start_segment": 0,
            "end_segment": expo_end,
            "start_time": segments[0].get("start", 0),
            "end_time": segments[expo_end].get("end", 0),
            "avg_intensity": round(sum(smoothed[:expo_end]) / expo_end, 3) if expo_end > 0 else 0,
            "description": "Initial emotional baseline and scene setup"
        })
    
    # Rising action: from exposition to climax
    if expo_end < climax_idx:
        phases.append({
            "phase": "rising_action",
            "start_segment": expo_end,
            "end_segment": climax_idx,
            "start_time": segments[expo_end].get("start", 0),
            "end_time": segments[climax_idx].get("start", 0),
            "avg_intensity": round(sum(smoothed[expo_end:climax_idx]) / (climax_idx - expo_end), 3),
            "description": "Building tension and emotional escalation"
        })
    
    # Climax: peak moment (1-2 segments around peak)
    climax_start = max(expo_end, climax_idx - 1)
    climax_end = min(total - 1, climax_idx + 1)
    phases.append({
        "phase": "climax",
        "start_segment": climax_start,
        "end_segment": climax_end,
        "start_time": segments[climax_start].get("start", 0),
        "end_time": segments[climax_end].get("end", 0),
        "peak_intensity": round(smoothed[climax_idx], 3),
        "peak_emotion": segments[climax_idx].get("fused_emotion"),
        "description": "Peak emotional intensity of the scene"
    })
    
    # Falling action: from climax to ~85%
    resolution_start = max(climax_end, int(total * 0.85))
    if climax_end < resolution_start:
        phases.append({
            "phase": "falling_action",
            "start_segment": climax_end,
            "end_segment": resolution_start,
            "start_time": segments[climax_end].get("start", 0),
            "end_time": segments[resolution_start].get("start", 0),
            "avg_intensity": round(sum(smoothed[climax_end:resolution_start]) / (resolution_start - climax_end), 3),
            "description": "Emotional tension decreasing"
        })
    
    # Resolution: final 15%
    if resolution_start < total:
        phases.append({
            "phase": "resolution",
            "start_segment": resolution_start,
            "end_segment": total - 1,
            "start_time": segments[resolution_start].get("start", 0),
            "end_time": segments[-1].get("end", 0),
            "avg_intensity": round(sum(smoothed[resolution_start:]) / (total - resolution_start), 3),
            "description": "Emotional resolution or new equilibrium"
        })
    
    return phases


def generate_arc_summary(
    segments: List[Dict[str, Any]],
    phases: List[Dict[str, Any]],
    turning_points: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of the emotional arc.
    """
    # Overall emotion distribution
    emotion_counts = defaultdict(int)
    for seg in segments:
        emo = seg.get("fused_emotion") or seg.get("speech_emotion")
        if emo:
            emotion_counts[emo] += 1
    
    total = sum(emotion_counts.values())
    emotion_distribution = {
        k: round(v / total, 3) for k, v in emotion_counts.items()
    } if total > 0 else {}
    
    # Dominant emotion overall
    dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
    
    # Calculate overall trajectory
    first_third = segments[:len(segments)//3]
    last_third = segments[-(len(segments)//3):]
    
    first_valence = sum(calculate_emotion_metrics(s)["valence"] for s in first_third) / len(first_third) if first_third else 0
    last_valence = sum(calculate_emotion_metrics(s)["valence"] for s in last_third) / len(last_third) if last_third else 0
    
    if last_valence > first_valence + 0.2:
        trajectory = "ascending"
        trajectory_desc = "Scene moves from negative/neutral to positive emotions"
    elif last_valence < first_valence - 0.2:
        trajectory = "descending"
        trajectory_desc = "Scene moves from positive/neutral to negative emotions"
    else:
        trajectory = "stable"
        trajectory_desc = "Emotional tone remains relatively consistent"
    
    # Conflict summary
    conflicts = [s for s in segments if s.get("conflict_type")]
    conflict_rate = len(conflicts) / len(segments) if segments else 0
    
    # Top turning points
    top_turning_points = turning_points[:3] if turning_points else []
    
    return {
        "total_segments": len(segments),
        "duration_seconds": segments[-1].get("end", 0) - segments[0].get("start", 0) if segments else 0,
        "dominant_emotion": dominant_emotion,
        "emotion_distribution": emotion_distribution,
        "trajectory": trajectory,
        "trajectory_description": trajectory_desc,
        "conflict_rate": round(conflict_rate, 3),
        "conflicts_detected": len(conflicts),
        "turning_points_count": len(turning_points),
        "key_turning_points": [
            {
                "time": f"{tp['timestamp']:.1f}s",
                "shift": f"{tp['from_emotion']} → {tp['to_emotion']}",
                "reasons": tp["reasons"]
            }
            for tp in top_turning_points
        ],
        "phases": [
            {
                "phase": p["phase"],
                "time_range": f"{p.get('start_time', 0):.1f}s - {p.get('end_time', 0):.1f}s",
                "description": p.get("description", "")
            }
            for p in phases
        ]
    }


def extract_emotion_arc(
    paired_data_path: str = "paired_data.json",
    output_path: str = "emotion_arc.json"
) -> Dict[str, Any]:
    """
    Main function to extract the complete emotional arc.
    
    Returns a comprehensive arc analysis including:
    - Emotional phases (exposition, rising action, climax, falling action, resolution)
    - Turning points with significance scores
    - Overall trajectory and summary
    """
    print("Loading paired data...")
    segments = load_paired_data(paired_data_path)
    
    print(f"Analyzing {len(segments)} segments...")
    
    # Detect turning points
    print("Detecting turning points...")
    turning_points = detect_turning_points(segments)
    print(f"  Found {len(turning_points)} turning points")
    
    # Identify arc phases
    print("Identifying arc phases...")
    phases = identify_arc_phases(segments)
    print(f"  Identified {len(phases)} phases")
    
    # Generate summary
    print("Generating arc summary...")
    summary = generate_arc_summary(segments, phases, turning_points)
    
    # Build complete result
    result = {
        "summary": summary,
        "phases": phases,
        "turning_points": turning_points,
        "segment_metrics": [
            {
                "index": i,
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": seg.get("text", "")[:50] + "..." if len(seg.get("text", "")) > 50 else seg.get("text", ""),
                **calculate_emotion_metrics(seg)
            }
            for i, seg in enumerate(segments)
        ]
    }
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved emotion arc to {output_path}")
    
    return result


def get_context_for_segment(
    arc_data: Dict[str, Any],
    segment_index: int
) -> Dict[str, Any]:
    """
    Get contextual information for a specific segment.
    Used by contextual_analyzer to provide arc context to GPT-4o.
    """
    phases = arc_data.get("phases", [])
    turning_points = arc_data.get("turning_points", [])
    summary = arc_data.get("summary", {})
    
    # Find which phase this segment is in
    current_phase = None
    for phase in phases:
        if phase.get("start_segment", 0) <= segment_index <= phase.get("end_segment", 999):
            current_phase = phase
            break
    
    # Find nearby turning points
    nearby_turning = [
        tp for tp in turning_points
        if abs(tp["segment_index"] - segment_index) <= 2
    ]
    
    # Progress through scene
    total_segments = summary.get("total_segments", 1)
    progress = segment_index / total_segments
    
    return {
        "current_phase": current_phase.get("phase") if current_phase else "unknown",
        "phase_description": current_phase.get("description") if current_phase else "",
        "scene_progress": round(progress, 2),
        "overall_trajectory": summary.get("trajectory"),
        "dominant_emotion": summary.get("dominant_emotion"),
        "is_near_turning_point": len(nearby_turning) > 0,
        "nearby_turning_points": nearby_turning
    }


def print_arc_summary(result: Dict[str, Any]):
    """Print a human-readable arc summary."""
    summary = result.get("summary", {})
    
    print("\n" + "=" * 60)
    print("EMOTIONAL ARC ANALYSIS")
    print("=" * 60)
    
    print(f"\nDuration: {summary.get('duration_seconds', 0):.1f}s")
    print(f"Segments: {summary.get('total_segments', 0)}")
    print(f"Dominant Emotion: {summary.get('dominant_emotion', 'unknown')}")
    print(f"Overall Trajectory: {summary.get('trajectory', 'unknown')}")
    print(f"  → {summary.get('trajectory_description', '')}")
    
    print(f"\nConflicts: {summary.get('conflicts_detected', 0)} ({summary.get('conflict_rate', 0)*100:.1f}% of segments)")
    
    print("\nPHASES:")
    for phase in summary.get("phases", []):
        print(f"  [{phase['time_range']}] {phase['phase'].upper()}")
        print(f"    {phase['description']}")
    
    print("\nKEY TURNING POINTS:")
    for tp in summary.get("key_turning_points", []):
        print(f"  [{tp['time']}] {tp['shift']}")
        print(f"    Reasons: {', '.join(tp['reasons'])}")


def main():
    """Run the emotion arc extraction."""
    result = extract_emotion_arc()
    print_arc_summary(result)


if __name__ == "__main__":
    main()
