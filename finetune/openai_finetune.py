"""
OpenAI Fine-Tuning Script for Scene Analysis

Fine-tunes GPT-4o-mini on your scene analysis task to create a specialized
model that understands movie scene interpretation with multimodal emotion data.

Prerequisites:
    pip install openai>=1.0.0

Usage:
    # Step 1: Create training data
    python create_training_data.py --format openai
    
    # Step 2: Upload and start fine-tuning
    python openai_finetune.py --train ./training_data/train.jsonl
    
    # Step 3: Check status
    python openai_finetune.py --status <job_id>
    
    # Step 4: Use fine-tuned model
    python openai_finetune.py --test <model_id>

Cost Estimate (as of 2024):
    - Training: ~$0.008 per 1K tokens
    - A typical scene analysis example is ~1000 tokens
    - 100 examples ≈ $0.80 for training
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai>=1.0.0")
    OpenAI = None


def get_client() -> OpenAI:
    """Get OpenAI client with API key."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


def validate_training_file(file_path: str) -> Dict[str, Any]:
    """
    Validate training file format before upload.
    Returns statistics about the file.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Training file not found: {file_path}")
    
    stats = {
        "total_examples": 0,
        "total_tokens_estimate": 0,
        "errors": []
    }
    
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            try:
                example = json.loads(line)
                
                # Check required structure
                if "messages" not in example:
                    stats["errors"].append(f"Line {i+1}: Missing 'messages' field")
                    continue
                
                messages = example["messages"]
                
                # Check for system, user, assistant messages
                roles = [m.get("role") for m in messages]
                if "system" not in roles:
                    stats["errors"].append(f"Line {i+1}: Missing system message (optional but recommended)")
                if "user" not in roles:
                    stats["errors"].append(f"Line {i+1}: Missing user message")
                if "assistant" not in roles:
                    stats["errors"].append(f"Line {i+1}: Missing assistant message")
                
                # Estimate tokens (rough: ~4 chars per token)
                total_chars = sum(len(m.get("content", "")) for m in messages)
                stats["total_tokens_estimate"] += total_chars // 4
                stats["total_examples"] += 1
                
            except json.JSONDecodeError:
                stats["errors"].append(f"Line {i+1}: Invalid JSON")
    
    return stats


def upload_training_file(client: OpenAI, file_path: str) -> str:
    """
    Upload training file to OpenAI.
    Returns the file ID.
    """
    print(f"Uploading {file_path}...")
    
    with open(file_path, "rb") as f:
        response = client.files.create(
            file=f,
            purpose="fine-tune"
        )
    
    print(f"File uploaded: {response.id}")
    return response.id


def create_fine_tune_job(
    client: OpenAI,
    training_file_id: str,
    validation_file_id: Optional[str] = None,
    model: str = "gpt-4o-mini-2024-07-18",
    suffix: str = "scene-analyzer",
    n_epochs: int = 3
) -> str:
    """
    Create a fine-tuning job.
    Returns the job ID.
    """
    print(f"\nCreating fine-tuning job...")
    print(f"  Base model: {model}")
    print(f"  Training file: {training_file_id}")
    print(f"  Epochs: {n_epochs}")
    
    params = {
        "training_file": training_file_id,
        "model": model,
        "suffix": suffix,
        "hyperparameters": {
            "n_epochs": n_epochs
        }
    }
    
    if validation_file_id:
        params["validation_file"] = validation_file_id
        print(f"  Validation file: {validation_file_id}")
    
    response = client.fine_tuning.jobs.create(**params)
    
    print(f"\nFine-tuning job created: {response.id}")
    print(f"Status: {response.status}")
    
    return response.id


def check_job_status(client: OpenAI, job_id: str) -> Dict[str, Any]:
    """Check the status of a fine-tuning job."""
    job = client.fine_tuning.jobs.retrieve(job_id)
    
    result = {
        "id": job.id,
        "status": job.status,
        "model": job.model,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
        "fine_tuned_model": job.fine_tuned_model,
        "trained_tokens": job.trained_tokens,
        "error": job.error
    }
    
    return result


def list_jobs(client: OpenAI, limit: int = 10) -> list:
    """List recent fine-tuning jobs."""
    jobs = client.fine_tuning.jobs.list(limit=limit)
    return [
        {
            "id": job.id,
            "status": job.status,
            "model": job.fine_tuned_model or job.model,
            "created_at": job.created_at
        }
        for job in jobs.data
    ]


def get_job_events(client: OpenAI, job_id: str, limit: int = 20) -> list:
    """Get events/logs for a fine-tuning job."""
    events = client.fine_tuning.jobs.list_events(
        fine_tuning_job_id=job_id,
        limit=limit
    )
    return [
        {
            "created_at": event.created_at,
            "level": event.level,
            "message": event.message
        }
        for event in events.data
    ]


def wait_for_completion(client: OpenAI, job_id: str, poll_interval: int = 60) -> Dict[str, Any]:
    """
    Wait for a fine-tuning job to complete.
    Polls status at regular intervals.
    """
    print(f"\nWaiting for job {job_id} to complete...")
    print(f"Polling every {poll_interval} seconds...\n")
    
    while True:
        status = check_job_status(client, job_id)
        print(f"[{time.strftime('%H:%M:%S')}] Status: {status['status']}")
        
        if status["status"] == "succeeded":
            print(f"\n✅ Fine-tuning complete!")
            print(f"Fine-tuned model: {status['fine_tuned_model']}")
            return status
        
        elif status["status"] == "failed":
            print(f"\n❌ Fine-tuning failed!")
            print(f"Error: {status['error']}")
            return status
        
        elif status["status"] == "cancelled":
            print(f"\n⚠️ Fine-tuning cancelled")
            return status
        
        time.sleep(poll_interval)


def test_model(client: OpenAI, model_id: str, test_prompt: str = None) -> str:
    """Test the fine-tuned model with a sample prompt."""
    
    if not test_prompt:
        test_prompt = """Analyze this movie scene based on the following transcript and emotion data:

[0.0s - 2.5s] "I can't believe you did that."
  Emotion: angry (confidence: 0.85)
[2.5s - 5.0s] "I had no choice. You know that."
  Emotion: sad (confidence: 0.72) [CONFLICT: voice=sad, face=neutral]
[5.0s - 8.0s] "There's always a choice."
  Emotion: neutral (confidence: 0.65)

Provide a comprehensive scene analysis including:
1. Scene overview (setting, mood, atmosphere)
2. Character emotional states and arcs
3. Thematic observations
4. Interpretation of any audio/visual emotion conflicts"""

    print(f"\nTesting model: {model_id}")
    print(f"\nPrompt:\n{test_prompt[:200]}...")
    
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are an expert film analyst specializing in multimodal scene interpretation."},
            {"role": "user", "content": test_prompt}
        ],
        max_tokens=1000,
        temperature=0.7
    )
    
    result = response.choices[0].message.content
    print(f"\nResponse:\n{result}")
    
    return result


def save_model_info(model_id: str, job_id: str, output_path: str = "finetune/model_info.json"):
    """Save fine-tuned model information for later use."""
    info = {
        "model_id": model_id,
        "job_id": job_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "usage": f"Use model_id '{model_id}' in scene_analyzer.py or contextual_analyzer.py"
    }
    
    with open(output_path, "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"Model info saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="OpenAI Fine-Tuning for Scene Analysis")
    
    # Actions
    parser.add_argument("--train", metavar="FILE", help="Start fine-tuning with training file")
    parser.add_argument("--validation", metavar="FILE", help="Validation file (optional)")
    parser.add_argument("--status", metavar="JOB_ID", help="Check status of a fine-tuning job")
    parser.add_argument("--wait", metavar="JOB_ID", help="Wait for job to complete")
    parser.add_argument("--events", metavar="JOB_ID", help="Get events/logs for a job")
    parser.add_argument("--list", action="store_true", help="List recent fine-tuning jobs")
    parser.add_argument("--test", metavar="MODEL_ID", help="Test a fine-tuned model")
    parser.add_argument("--cancel", metavar="JOB_ID", help="Cancel a fine-tuning job")
    
    # Options
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18", help="Base model")
    parser.add_argument("--suffix", default="scene-analyzer", help="Model suffix")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    
    args = parser.parse_args()
    
    if not OpenAI:
        print("OpenAI package not installed. Run: pip install openai>=1.0.0")
        return
    
    client = get_client()
    
    # Handle actions
    if args.train:
        # Validate training file
        print("Validating training file...")
        stats = validate_training_file(args.train)
        
        print(f"\nTraining file statistics:")
        print(f"  Examples: {stats['total_examples']}")
        print(f"  Estimated tokens: {stats['total_tokens_estimate']:,}")
        print(f"  Estimated cost: ${stats['total_tokens_estimate'] * 0.008 / 1000:.2f}")
        
        if stats["errors"]:
            print(f"\n⚠️ Validation warnings ({len(stats['errors'])}):")
            for err in stats["errors"][:5]:
                print(f"  - {err}")
            if len(stats["errors"]) > 5:
                print(f"  ... and {len(stats['errors']) - 5} more")
        
        # Upload training file
        training_file_id = upload_training_file(client, args.train)
        
        # Upload validation file if provided
        validation_file_id = None
        if args.validation:
            validation_file_id = upload_training_file(client, args.validation)
        
        # Create fine-tuning job
        job_id = create_fine_tune_job(
            client,
            training_file_id,
            validation_file_id,
            model=args.model,
            suffix=args.suffix,
            n_epochs=args.epochs
        )
        
        print(f"\n📋 To check status: python openai_finetune.py --status {job_id}")
        print(f"📋 To wait for completion: python openai_finetune.py --wait {job_id}")
    
    elif args.status:
        status = check_job_status(client, args.status)
        print(f"\nJob Status:")
        for key, value in status.items():
            if value is not None:
                print(f"  {key}: {value}")
    
    elif args.wait:
        status = wait_for_completion(client, args.wait)
        if status["fine_tuned_model"]:
            save_model_info(status["fine_tuned_model"], args.wait)
    
    elif args.events:
        events = get_job_events(client, args.events)
        print(f"\nRecent events for job {args.events}:")
        for event in events:
            print(f"  [{event['level']}] {event['message']}")
    
    elif args.list:
        jobs = list_jobs(client)
        print(f"\nRecent fine-tuning jobs:")
        for job in jobs:
            print(f"  {job['id']} | {job['status']} | {job['model']}")
    
    elif args.test:
        test_model(client, args.test)
    
    elif args.cancel:
        client.fine_tuning.jobs.cancel(args.cancel)
        print(f"Cancelled job: {args.cancel}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
