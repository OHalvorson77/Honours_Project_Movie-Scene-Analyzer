import json
import os
from collections import Counter


# This function is a helper function used to come to a conclusion on dominant emotion for a segment from the emotions from the frames
def aggregate_facial_emotions(frame_emotions: list) -> dict:

    if not frame_emotions:
        return {
            "dominant_emotion": None,
            "confidence": 0,
            "distribution": {},
            "faces_detected": 0,
            "total_frames": 0
        }
    
    # Count frames with detected faces
    detected_frames = [f for f in frame_emotions if f.get("face_detected")]
    
    if not detected_frames:
        return {
            "dominant_emotion": None,
            "confidence": 0,
            "distribution": {},
            "faces_detected": 0,
            "total_frames": len(frame_emotions)
        }
    
    # Aggregate emotion scores across all frames
    emotion_totals = {}
    for frame in detected_frames:
        for emotion, score in frame.get("all_emotions", {}).items():
            emotion_totals[emotion] = emotion_totals.get(emotion, 0) + score
    
    # Average the scores
    num_frames = len(detected_frames)
    emotion_avg = {k: round(v / num_frames, 3) for k, v in emotion_totals.items()}
    
    # Find dominant emotion
    dominant = max(emotion_avg, key=emotion_avg.get) if emotion_avg else None
    confidence = emotion_avg.get(dominant, 0) if dominant else 0
    
    # Create am object to return with various fields from the aggregated emotions
    return {
        "dominant_emotion": dominant,
        "confidence": round(confidence, 3),
        "distribution": emotion_avg,
        "faces_detected": len(detected_frames),
        "total_frames": len(frame_emotions)
    }


def blend_emotions(speech_emotion: dict, face_emotion: dict) -> dict:

    speech_emo = speech_emotion.get("emotion")
    speech_conf = speech_emotion.get("confidence", 0) or 0
    
    face_emo = face_emotion.get("dominant_emotion")
    face_conf = face_emotion.get("confidence", 0) or 0
    
    # Emotion mapping (DeepFace and wav2vec2 use slightly different labels)
    # DeepFace: angry, disgust, fear, happy, sad, surprise, neutral
    # Wav2Vec2: angry, calm, disgust, fear, happy, neutral, sad, surprise
    emotion_map = {
        "calm": "neutral",  # Map calm to neutral for comparison since then it makes models the same 
    }
    
    speech_emo_normalized = emotion_map.get(speech_emo, speech_emo)
    face_emo_normalized = emotion_map.get(face_emo, face_emo)
    
    # Determine if the emotions agree or disagree (None if no emotion is detected)
    if speech_emo_normalized and face_emo_normalized:
        emotions_match = speech_emo_normalized == face_emo_normalized
    else:
        emotions_match = None  
    
    # Fused emotion: weighted by confidence
    if speech_emo and face_emo:
        total_conf = speech_conf + face_conf
        if total_conf > 0:
            # Higher confidence wins
            if speech_conf >= face_conf:
                fused_emotion = speech_emo
                fused_confidence = speech_conf
                fused_source = "speech"
            else:
                fused_emotion = face_emo
                fused_confidence = face_conf
                fused_source = "face"
        else:
            fused_emotion = speech_emo
            fused_confidence = 0
            fused_source = "speech"
    elif speech_emo:
        fused_emotion = speech_emo
        fused_confidence = speech_conf
        fused_source = "speech"
    elif face_emo:
        fused_emotion = face_emo
        fused_confidence = face_conf
        fused_source = "face"
    else:
        fused_emotion = None
        fused_confidence = 0
        fused_source = None
    
    # Detect particular conflicts for LLM interpretation
    conflict_type = None
    if speech_emo and face_emo and not emotions_match:
        # Flag specific conflicts types that could be secret undetected expressions
        conflict_pairs = {
            ("calm", "angry"): "possible_sarcasm",
            ("neutral", "angry"): "possible_sarcasm",
            ("happy", "sad"): "masked_emotion",
            ("calm", "fear"): "suppressed_fear",
            ("neutral", "fear"): "suppressed_fear",
            ("happy", "angry"): "passive_aggressive",
            ("calm", "sad"): "hidden_sadness",
        }
        pair = (speech_emo_normalized, face_emo_normalized)
        reverse_pair = (face_emo_normalized, speech_emo_normalized)
        conflict_type = conflict_pairs.get(pair) or conflict_pairs.get(reverse_pair) or "emotion_mismatch"
    
    return {
        "fused_emotion": fused_emotion,
        "fused_confidence": round(fused_confidence, 3) if fused_confidence else 0,
        "fused_source": fused_source,
        "emotions_match": emotions_match,
        "conflict_type": conflict_type
    }


def pair_frames_to_transcript(
    transcript_path: str = "transcript.json",
    frames_path: str = "frames.json",
    emotions_path: str = "emotions.json",
    face_emotions_path: str = "face_emotions.json",
    output_path: str = "paired_data.json"
):

    # Loading in all the available data from the previous steps
    with open(frames_path, "r") as f:
        frames_data = json.load(f)
    
    with open(transcript_path, "r") as f:
        transcript_data = json.load(f)

    emotions_data = []
    if os.path.exists(emotions_path):
        with open(emotions_path, "r") as f:
            emotions_data = json.load(f)

    # Load facial emotions if available
    face_emotions_data = []
    if os.path.exists(face_emotions_path):
        with open(face_emotions_path, "r") as f:
            face_emotions_data = json.load(f)
    
    # Creating lookups for the speech and face emotions
    emotion_lookup = {e["start"]: e for e in emotions_data}
    face_emotion_lookup = {f["filename"]: f for f in face_emotions_data}

    frames = frames_data["frames"]
    
    paired = []
    for segment in transcript_data:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]

        # Find all frames within this segment's time range
        matching_frames = [
            frame for frame in frames
            if start <= frame["timestamp"] < end
        ]

        # Get speech emotion data for this segment
        speech_emotion = emotion_lookup.get(start, {})
        
        # Get facial emotions for matching frames
        frame_face_emotions = [
            face_emotion_lookup.get(f["filename"], {})
            for f in matching_frames
        ]
        
        # Aggregate facial emotions across frames
        aggregated_face = aggregate_facial_emotions(frame_face_emotions)
        
        # Blend speech and facial emotions
        blended = blend_emotions(speech_emotion, aggregated_face)
        
        # Adding all the results for the segment to the paired array
        paired.append({
            "start": start,
            "end": end,
            "text": text,
            # Speech emotion
            "speech_emotion": speech_emotion.get("emotion"),
            "speech_confidence": speech_emotion.get("confidence"),
            # Facial emotion (aggregated)
            "face_emotion": aggregated_face["dominant_emotion"],
            "face_confidence": aggregated_face["confidence"],
            "face_distribution": aggregated_face["distribution"],
            "faces_detected": aggregated_face["faces_detected"],
            # Blended/fused result
            "fused_emotion": blended["fused_emotion"],
            "fused_confidence": blended["fused_confidence"],
            "fused_source": blended["fused_source"],
            "emotions_match": blended["emotions_match"],
            "conflict_type": blended["conflict_type"],
            # Frames
            "frames": matching_frames
        })

    # Saving the paired data to a json file
    with open(output_path, "w") as f:
        json.dump(paired, f, indent=2)

    # Print summary of the pairing findings to the console
    print(f"Paired {len(frames)} frames across {len(transcript_data)} transcript segments\n")
    
    conflicts = [p for p in paired if p["conflict_type"]]
    print(f"Emotion conflicts detected: {len(conflicts)}/{len(paired)} segments\n")
    

    print(f"Paired data saved to {output_path}")
    return paired


if __name__ == "__main__":
    pair_frames_to_transcript()
