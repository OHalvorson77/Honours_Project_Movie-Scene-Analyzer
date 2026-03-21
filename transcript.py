from faster_whisper import WhisperModel
import json

def transcribe_video(video_path: str, output_path: str = "transcript.json"):
    model = WhisperModel("medium", device="cpu")
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
