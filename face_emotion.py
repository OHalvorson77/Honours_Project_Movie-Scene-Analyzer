import json
import os
from collections import Counter
from deepface import DeepFace

def analyze_facial_emotions(
    frames_dir: str = "frames",
    output_path: str = "face_emotions.json"
):
    """
    Run facial emotion detection on all extracted frames using DeepFace.
    """
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    print(f"Analyzing facial emotions for {len(frames)} frames...")
    
    results = []
    
    for i, frame in enumerate(frames):
        frame_path = os.path.join(frames_dir, frame)
        
        try:
            # DeepFace analyze returns emotion predictions
            analysis = DeepFace.analyze(
                frame_path,
                actions=['emotion'],
                enforce_detection=False,  # Don't fail if no face detected
                silent=True
            )
            
            # DeepFace returns a list if multiple faces, take the first/dominant
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            emotions = analysis.get('emotion', {})
            dominant_emotion = analysis.get('dominant_emotion', 'unknown')
            
            # Normalize confidence scores to 0-1 range (DeepFace returns 0-100)
            # Convert to Python float to ensure JSON serializable
            emotions_normalized = {k: round(float(v) / 100, 3) for k, v in emotions.items()}
            
            results.append({
                "filename": frame,
                "face_detected": True,
                "dominant_emotion": dominant_emotion,
                "confidence": float(emotions_normalized.get(dominant_emotion, 0)),
                "all_emotions": emotions_normalized
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(frames)} frames...")
                
        except Exception as e:
            if i < 3:
                print(f"  Frame {frame} error: {e}")
            results.append({
                "filename": frame,
                "face_detected": False,
                "dominant_emotion": None,
                "confidence": 0,
                "all_emotions": {}
            })
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary stats
    detected = sum(1 for r in results if r["face_detected"])
    print(f"\nFacial emotion analysis complete!")
    print(f"  Faces detected: {detected}/{len(frames)} frames")
    print(f"  Results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    analyze_facial_emotions()

