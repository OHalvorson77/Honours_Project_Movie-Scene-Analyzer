import subprocess
import os
import json


# Function made to extract the frames from video into a json file
def extract_frames(
    video_path: str,
    output_dir: str = "frames",
    interval_seconds: float = 0.5,
    metadata_path: str = "frames.json"
):

    # Creating the new dir for output of the frames
    os.makedirs(output_dir, exist_ok=True)

    # Using ffmpeg I extract the frames at a fixed rate deaulted to 0.5 seconds from function parameters
    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"fps=1/{interval_seconds}",
        f"{output_dir}/frame_%04d.jpg"
    ]
    subprocess.run(command, check=True)

    # Build frame metadata with timestamps from the saved frames folder created above
    frames = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
    
    # Calculating the timestamp by checking for the frame number and then fultiplying it by interval seconds, also saving the path to the frame image
    frame_data = [
        {
            "filename": frame,
            "path": os.path.join(output_dir, frame),
            "timestamp": i * interval_seconds
        }
        for i, frame in enumerate(frames)
    ]

    # Creating a frames.json file and saving the frame data gathered above to it
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
