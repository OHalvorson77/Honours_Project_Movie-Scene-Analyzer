from faster_whisper import WhisperModel
import json

def transcribe_video(video_path: str, output_path: str = "transcript.json", model_size: str = "base"):
    print(f"Loading Whisper model ({model_size})...", flush=True)

    # Loading in the whisper model which comes from faster-whisper
    model = WhisperModel(model_size, device="cpu")

    # Calling the transcribe function from faster whisper to transcribe the video
    segments = model.transcribe(video_path)

    transcript_data = []

    # Iterating over the segments and adding to the transript_data array the text, and start + end timestamps of each segment
    for segment in segments:
        transcript_data.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })

    # Saving the transcript data to a json file (transcript.json as default from the function parameters)
    with open(output_path, "w") as f:
        json.dump(transcript_data, f, indent=2)

    print(f"\nVideoranscript saved to {output_path}")
    return transcript_data

if __name__ == "__main__":
    transcribe_video("scene.mp4")
