"""
FastAPI Backend for Movie Scene Analyzer

Provides endpoints for:
- Video upload (max 5 minutes)
- Pipeline execution with progress tracking
- Retrieving analysis results
"""

import os
import json
import shutil
import asyncio
import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Scene Analyzer API", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Job tracking
jobs: dict[str, dict] = {}

# Max video duration (5 minutes in seconds)
MAX_DURATION_SECONDS = 300


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: int  # 0-100
    current_step: str
    error: Optional[str] = None
    created_at: str


def get_video_duration(video_path: str) -> float:
    """Get video duration using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path
            ],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0


async def run_pipeline_async(job_id: str, video_path: str, output_dir: str):
    """Run the analysis pipeline asynchronously."""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["current_step"] = "Initializing..."
        jobs[job_id]["progress"] = 5
        
        # Step 1: Transcription (0-20%)
        jobs[job_id]["current_step"] = "Transcribing audio..."
        jobs[job_id]["progress"] = 10
        
        process = await asyncio.create_subprocess_exec(
            "python", "-c", f"""
import sys
sys.path.insert(0, '.')
from transcript import transcribe_video
transcribe_video('{video_path}', '{output_dir}/transcript.json')
""",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.wait()
        if process.returncode != 0:
            raise Exception("Transcription failed")
        jobs[job_id]["progress"] = 20
        
        # Step 2: Frame extraction (20-35%)
        jobs[job_id]["current_step"] = "Extracting frames..."
        frames_dir = f"{output_dir}/frames"
        os.makedirs(frames_dir, exist_ok=True)
        
        process = await asyncio.create_subprocess_exec(
            "python", "-c", f"""
import sys
sys.path.insert(0, '.')
from frames import extract_frames
extract_frames('{video_path}', '{frames_dir}', 0.5, '{output_dir}/frames.json')
""",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.wait()
        if process.returncode != 0:
            raise Exception("Frame extraction failed")
        jobs[job_id]["progress"] = 35
        
        # Step 3: Speech emotion (35-50%)
        jobs[job_id]["current_step"] = "Analyzing speech emotions..."
        
        process = await asyncio.create_subprocess_exec(
            "python", "-c", f"""
import sys
sys.path.insert(0, '.')
from emotion import classify_speech_emotions
classify_speech_emotions('{video_path}', '{output_dir}/transcript.json', '{output_dir}/emotions.json')
""",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.wait()
        if process.returncode != 0:
            raise Exception("Speech emotion analysis failed")
        jobs[job_id]["progress"] = 50
        
        # Step 4: Facial emotion (50-70%)
        jobs[job_id]["current_step"] = "Analyzing facial expressions..."
        
        process = await asyncio.create_subprocess_exec(
            "python", "-c", f"""
import sys
sys.path.insert(0, '.')
from face_emotion import analyze_facial_emotions
analyze_facial_emotions('{frames_dir}', '{output_dir}/face_emotions.json')
""",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.wait()
        if process.returncode != 0:
            raise Exception("Facial emotion analysis failed")
        jobs[job_id]["progress"] = 70
        
        # Step 5: Data pairing (70-80%)
        jobs[job_id]["current_step"] = "Fusing multimodal data..."
        
        process = await asyncio.create_subprocess_exec(
            "python", "-c", f"""
import sys
sys.path.insert(0, '.')
from pair import pair_frames_to_transcript
pair_frames_to_transcript(
    '{output_dir}/transcript.json',
    '{output_dir}/frames.json', 
    '{output_dir}/emotions.json',
    '{output_dir}/face_emotions.json',
    '{output_dir}/paired_data.json'
)
""",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.wait()
        if process.returncode != 0:
            raise Exception("Data pairing failed")
        jobs[job_id]["progress"] = 80
        
        # Step 6: Keyframe extraction (80-85%)
        jobs[job_id]["current_step"] = "Selecting keyframes..."
        
        process = await asyncio.create_subprocess_exec(
            "python", "-c", f"""
import sys
sys.path.insert(0, '.')
import json
from keyframe_extractor import extract_keyframes, get_conflict_summary

with open('{output_dir}/paired_data.json') as f:
    paired_data = json.load(f)

keyframes = extract_keyframes(paired_data)
with open('{output_dir}/keyframes.json', 'w') as f:
    json.dump(keyframes, f, indent=2)

conflict_summary = get_conflict_summary(paired_data)
with open('{output_dir}/conflict_summary.json', 'w') as f:
    json.dump(conflict_summary, f, indent=2)
""",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.wait()
        jobs[job_id]["progress"] = 85
        
        # Step 7: GPT-4o Analysis (85-100%) - Optional
        if os.environ.get("OPENAI_API_KEY"):
            jobs[job_id]["current_step"] = "Generating AI scene analysis..."
            
            process = await asyncio.create_subprocess_exec(
                "python", "-c", f"""
import sys
sys.path.insert(0, '.')
import json
from scene_analyzer import analyze_scene, load_conflict_summary

with open('{output_dir}/keyframes.json') as f:
    keyframes = json.load(f)
with open('{output_dir}/conflict_summary.json') as f:
    conflict_summary = json.load(f)

# Update frame paths for the output directory
for kf in keyframes:
    kf['frame']['path'] = '{output_dir}/frames/' + kf['frame']['filename']

analysis = analyze_scene(keyframes, conflict_summary, max_frames=8)
with open('{output_dir}/scene_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)
""",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.wait()
        else:
            # Create placeholder analysis
            placeholder = {
                "scene_overview": {
                    "setting": "Analysis requires OPENAI_API_KEY",
                    "time_of_day": "N/A",
                    "atmosphere": "Set OPENAI_API_KEY environment variable for AI analysis",
                    "mood": "N/A"
                },
                "characters": [],
                "cinematic_elements": {
                    "camera_work": "N/A",
                    "lighting": "N/A",
                    "color_palette": "N/A",
                    "notable_techniques": "N/A"
                },
                "emotional_arc": {"progression": "N/A", "key_moments": []},
                "conflict_interpretations": [],
                "themes": {"central_themes": [], "narrative_significance": "N/A"},
                "summary": "AI scene analysis not available. Set OPENAI_API_KEY for GPT-4o analysis."
            }
            with open(f"{output_dir}/scene_analysis.json", "w") as f:
                json.dump(placeholder, f, indent=2)
        
        jobs[job_id]["progress"] = 100
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["current_step"] = "Analysis complete!"
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["current_step"] = f"Error: {str(e)}"


@app.post("/api/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a video file and start processing."""
    
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        raise HTTPException(status_code=400, detail="Only video files (mp4, mov, avi, mkv) are allowed")
    
    # Generate job ID
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    
    # Save uploaded file
    video_path = job_dir / "video.mp4"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Check video duration
    duration = get_video_duration(str(video_path))
    if duration > MAX_DURATION_SECONDS:
        shutil.rmtree(job_dir)
        raise HTTPException(
            status_code=400, 
            detail=f"Video too long ({duration:.1f}s). Maximum duration is 5 minutes (300s)."
        )
    
    # Initialize job
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "current_step": "Queued for processing",
        "error": None,
        "created_at": datetime.now().isoformat(),
        "video_path": str(video_path),
        "output_dir": str(job_dir),
        "duration": duration
    }
    
    # Start processing in background
    background_tasks.add_task(run_pipeline_async, job_id, str(video_path), str(job_dir))
    
    return {"job_id": job_id, "message": "Upload successful, processing started"}


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Get the analysis results for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job['status']}")
    
    output_dir = Path(job["output_dir"])
    
    # Load all result files
    results = {}
    
    try:
        with open(output_dir / "paired_data.json") as f:
            results["paired_data"] = json.load(f)
    except FileNotFoundError:
        results["paired_data"] = []
    
    try:
        with open(output_dir / "scene_analysis.json") as f:
            results["scene_analysis"] = json.load(f)
    except FileNotFoundError:
        results["scene_analysis"] = None
    
    try:
        with open(output_dir / "conflict_summary.json") as f:
            results["conflict_summary"] = json.load(f)
    except FileNotFoundError:
        results["conflict_summary"] = None
    
    return results


@app.get("/api/video/{job_id}")
async def get_video(job_id: str):
    """Serve the uploaded video file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    video_path = Path(jobs[job_id]["video_path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs."""
    return list(jobs.values())


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    output_dir = Path(job["output_dir"])
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    del jobs[job_id]
    return {"message": "Job deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



