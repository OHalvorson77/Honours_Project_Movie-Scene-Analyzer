import json
import os
from collections import Counter
from deepface import DeepFace

def analyze_facial_emotions(
    frames_dir: str = "frames",
    output_path: str = "face_emotions.json"
):

    # Gather all of the frames from the frames directory into a sorted list
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    print(f"Analyzing facial emotions for {len(frames)} frames...")
    
    results = []
    
    # Iterating over enumerated frames and then joining the frames directory with the frame name to get the full path to the frame
    for i, frame in enumerate(frames):
        frame_path = os.path.join(frames_dir, frame)
        

        # Run Deepface model with analyze function to get the emotion predictions
        try:
            # DeepFace analyze returns emotion predictions
            analysis = DeepFace.analyze(
                frame_path,
                actions=['emotion'],
                enforce_detection=False,  # Don't fail if no face detected
                silent=True
            )
            
            # Checks if DeepFace returns a face, take the first/dominant since it might reurn multiple
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            emotions = analysis.get('emotion', {})
            dominant_emotion = analysis.get('dominant_emotion', 'unknown')
            
            # Normalize from deepface 0-100 to 0-1 range and convert to python float
            emotions_normalized = {k: round(float(v) / 100, 3) for k, v in emotions.items()}
            
            # For all face detected frames, add the results to the results list for the various fields (Dominant emotion, confidence, and all emotions)
            results.append({
                "filename": frame,
                "face_detected": True,
                "dominant_emotion": dominant_emotion,
                "confidence": float(emotions_normalized.get(dominant_emotion, 0)),
                "all_emotions": emotions_normalized
            })
            

        # For any errors or no face detected frames, add the results to the results list with null values and False for face detected       
        except Exception as e:
            if i < 3:
                print(f"  Frame {frame} error: {str(e).encode('ascii', 'replace').decode()}")
            results.append({
                "filename": frame,
                "face_detected": False,
                "dominant_emotion": None,
                "confidence": 0,
                "all_emotions": {}
            })

    # Saving the results to a json file
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    analyze_facial_emotions()

