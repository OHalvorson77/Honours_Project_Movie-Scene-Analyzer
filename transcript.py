from faster_whisper import WhisperModel
import json
import sys

def transcribe_video(video_path: str, output_path: str = "transcript.json", model_size: str = "base"):
    print(f"Loading Whisper model ({model_size})...", flush=True)
    model = WhisperModel(model_size, device="cpu")
    print(f"Transcribing {video_path}...", flush=True)
    segments, info = model.transcribe(video_path)

    transcript_data = []
    for segment in segments:
        transcript_data.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })

    with open(output_path, "w") as f:
        json.dump(transcript_data, f, indent=2)

    print(f"\nTranscript saved to {output_path}")
    return transcript_data

if __name__ == "__main__":
    transcribe_video("scene.mp4")
