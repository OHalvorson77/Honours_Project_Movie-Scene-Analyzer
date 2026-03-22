import json
import subprocess
import os
import torch
from transformers import pipeline

def extract_audio_segment(video_path: str, start: float, end: float, output_path: str):
    """Extract audio segment from video using ffmpeg."""
    duration = end - start
    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", str(start),
        "-t", str(duration),
        "-vn",  # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",  # 16kHz for speech models
        "-ac", "1",  # mono
        output_path
    ]
    subprocess.run(command, capture_output=True, check=True)

def classify_speech_emotions(
    video_path: str = "scene.mp4",
    transcript_path: str = "transcript.json",
    output_path: str = "emotions.json",
    temp_dir: str = "temp_audio"
):
    """
    Run speech emotion recognition on each transcript segment.
    """
    os.makedirs(temp_dir, exist_ok=True)
    
    with open(transcript_path, "r") as f:
        transcript = json.load(f)
    
    print("Loading speech emotion recognition model...")
    classifier = pipeline(
        "audio-classification",
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        device=0 if torch.cuda.is_available() else -1
    )
    
    results = []
    
    for i, segment in enumerate(transcript):
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        
        print(f"Processing segment {i+1}/{len(transcript)}: [{start:.2f}s - {end:.2f}s]")
        
        audio_path = os.path.join(temp_dir, f"segment_{i:04d}.wav")
        try:
            extract_audio_segment(video_path, start, end, audio_path)
            
            predictions = classifier(audio_path)
            
            top_emotion = predictions[0]
            all_emotions = {p["label"]: round(p["score"], 3) for p in predictions}
            
            results.append({
                "start": start,
                "end": end,
                "text": text,
                "emotion": top_emotion["label"],
                "confidence": round(top_emotion["score"], 3),
                "all_emotions": all_emotions
            })
            
            print(f"  -> {top_emotion['label']} ({top_emotion['score']:.2%})")
            
        except Exception as e:
            print(f"  -> Error processing segment: {e}")
            results.append({
                "start": start,
                "end": end,
                "text": text,
                "emotion": "unknown",
                "confidence": 0.0,
                "all_emotions": {}
            })
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)
    
    print(f"\nEmotion annotations saved to {output_path}")
    return results

if __name__ == "__main__":
    classify_speech_emotions()

