import subprocess
import os
import json

def extract_frames(
    video_path: str,
    output_dir: str = "frames",
    interval_seconds: float = 0.5,
    metadata_path: str = "frames.json"
):
    os.makedirs(output_dir, exist_ok=True)

    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"fps=1/{interval_seconds}",
        f"{output_dir}/frame_%04d.jpg"
    ]
    subprocess.run(command, check=True)

    # Build frame metadata with timestamps
    frames = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
    
    frame_data = [
        {
            "filename": frame,
            "path": os.path.join(output_dir, frame),
            "timestamp": i * interval_seconds
        }
        for i, frame in enumerate(frames)
    ]

    with open(metadata_path, "w") as f:
        json.dump({
            "interval_seconds": interval_seconds,
            "frames": frame_data
        }, f, indent=2)

    print(f"Extracted {len(frames)} frames to {output_dir}/")
    print(f"Frame metadata saved to {metadata_path}")
    return frame_data

if __name__ == "__main__":
    extract_frames("scene.mp4", "frames", interval_seconds=0.5)
