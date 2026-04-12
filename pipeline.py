import os
import json
from transcript import transcribe_video
from frames import extract_frames
from emotion import classify_speech_emotions
from face_emotion import analyze_facial_emotions
from pair import pair_frames_to_transcript
from keyframe_extractor import extract_keyframes, load_paired_data, save_keyframes, get_conflict_summary
from scene_analyzer import analyze_scene, load_keyframes as load_kf, load_conflict_summary, save_analysis
from claude_analyzer import analyze_scene_claude, save_analysis as save_claude_analysis
from compare_analyses import compare_analyses, save_comparison, print_summary
from emotion_arc import extract_emotion_arc, print_arc_summary
from contextual_analyzer import run_contextual_analysis
from narrative_refiner import refine_narrative, print_refined_summary

# Fine-tuned model support
try:
    from finetune import (
        print_model_status,
        analyze_scene_finetuned,
        get_finetuned_gpt_model_id
    )
    HAS_FINETUNE = True
except ImportError:
    HAS_FINETUNE = False


def run_phase1(
    video_path: str = "scene.mp4",
    frames_dir: str = "frames",
    interval_seconds: float = 0.5
):
    # Runs the first phase of the pipeline which is the media parsing and emotion extraction
    print("=" * 60)
    print("PHASE 1: CORE PIPELINE")
    print("=" * 60)
    
    print("\n" + "-" * 60)
    print("STEP 1.1: Transcribing video...")
    print("-" * 60)
    transcribe_video(video_path)

    print("\n" + "-" * 60)
    print("STEP 1.2: Extracting frames...")
    print("-" * 60)
    extract_frames(video_path, frames_dir, interval_seconds)

    print("\n" + "-" * 60)
    print("STEP 1.3: Speech emotion classification...")
    print("-" * 60)
    classify_speech_emotions(video_path)

    print("\n" + "-" * 60)
    print("STEP 1.4: Facial emotion detection (DeepFace)...")
    print("-" * 60)
    analyze_facial_emotions(frames_dir)

    print("\n" + "-" * 60)
    print("STEP 1.5: Pairing & fusing emotions...")
    print("-" * 60)
    pair_frames_to_transcript()

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE!")
    print("=" * 60)
    print("\nOutputs:")
    print("  - transcript.json     (timestamped transcript)")
    print("  - frames.json         (frame metadata)")
    print("  - emotions.json       (speech emotions per line)")
    print("  - face_emotions.json  (facial emotions per frame)")
    print("  - paired_data.json    (combined + fused emotions)")


def run_phase2(max_frames: int = 10, skip_claude: bool = False, use_finetuned: bool = False):
    print("\n" + "=" * 60)
    print("PHASE 2: SEMANTIC UNDERSTANDING")
    print("=" * 60)
    
    # Chekcs if the finetuned models exist
    if use_finetuned and HAS_FINETUNE:
        print_model_status()
    
    print("\n" + "-" * 60)
    print("STEP 2.1: Extracting keyframes...")
    print("-" * 60)
    paired_data = load_paired_data()
    keyframes = extract_keyframes(paired_data)
    save_keyframes(keyframes)
    
    # Save conflict summary
    conflict_summary = get_conflict_summary(paired_data)
    with open("conflict_summary.json", "w") as f:
        json.dump(conflict_summary, f, indent=2)
    print(f"Saved conflict summary ({conflict_summary['conflict_count']} conflicts)")
    
    # Load keyframes for analysis
    keyframes = load_kf()
    conflict_summary = load_conflict_summary()
    
    # Step 2.2: GPT-4o Analysis (or fine-tuned model)
    print("\n" + "-" * 60)
    if use_finetuned and HAS_FINETUNE and get_finetuned_gpt_model_id():
        print("STEP 2.2: Fine-tuned model scene analysis...")
    else:
        print("STEP 2.2: GPT-4o Vision scene analysis...")
    print("-" * 60)
    
    gpt4o_done = False
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set - skipping GPT-4o analysis")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
    else:
        # Use fine-tuned model if available and requested
        if use_finetuned and HAS_FINETUNE:
            result = analyze_scene_finetuned(keyframes, conflict_summary)
            analysis = {"analysis": result["analysis"], "_metadata": {"model": result["model"]}}
            save_analysis(analysis)
            print(f"✓ Analysis complete (model: {result['model']})")
        else:
            analysis = analyze_scene(keyframes, conflict_summary, max_frames=max_frames)
            save_analysis(analysis)
            print("✓ GPT-4o analysis complete")
        gpt4o_done = True
    
    # Step 2.3: Claude Analysis (cross-validation)
    print("\n" + "-" * 60)
    print("STEP 2.3: Claude cross-validation...")
    print("-" * 60)
    
    claude_done = False
    if skip_claude:
        print("Skipping Claude analysis (--skip-claude flag)")
    elif not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n⚠️  ANTHROPIC_API_KEY not set - skipping Claude analysis")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
    else:
        claude_analysis = analyze_scene_claude(keyframes, conflict_summary, max_frames=max_frames)
        save_claude_analysis(claude_analysis)
        claude_done = True
        print("✓ Claude analysis complete")
    
    # Step 2.4: Compare analyses
    if gpt4o_done and claude_done:
        print("\n" + "-" * 60)
        print("STEP 2.4: Cross-validation comparison...")
        print("-" * 60)
        comparison = compare_analyses()
        save_comparison(comparison)
        print_summary(comparison)
    elif gpt4o_done or claude_done:
        print("\n⚠️  Only one model ran - skipping comparison")
        print("Set both API keys for full cross-validation")
    
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE!")
    print("=" * 60)
    print("\nOutputs:")
    print("  - keyframes.json         (selected emotionally significant frames)")
    print("  - conflict_summary.json  (audio/visual emotion mismatches)")
    if gpt4o_done:
        print("  - scene_analysis.json    (GPT-4o narrative analysis)")
    if claude_done:
        print("  - claude_analysis.json   (Claude narrative analysis)")
    if gpt4o_done and claude_done:
        print("  - analysis_comparison.json (cross-validation results)")


def run_phase2_5(max_frames: int = 10):
    print("\n" + "=" * 60)
    print("PHASE 2.5: CONTEXTUAL GROUNDING")
    print("=" * 60)
    
    # Step 2.5.1: Extract emotional arc
    print("\n" + "-" * 60)
    print("STEP 2.5.1: Extracting emotional arc...")
    print("-" * 60)
    arc_result = extract_emotion_arc()
    print_arc_summary(arc_result)
    
    # Step 2.5.2: Contextual analysis with sliding window
    print("\n" + "-" * 60)
    print("STEP 2.5.2: Contextual analysis (sliding window)...")
    print("-" * 60)
    
    contextual_done = False
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set - skipping contextual analysis")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
    else:
        run_contextual_analysis(max_frames=max_frames)
        contextual_done = True
        print("✓ Contextual analysis complete")
    
    # Step 2.5.3: Narrative refinement
    print("\n" + "-" * 60)
    print("STEP 2.5.3: Narrative refinement...")
    print("-" * 60)
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set - merging without LLM refinement")
    
    refined = refine_narrative()
    print_refined_summary(refined)
    
    print("\n" + "=" * 60)
    print("PHASE 2.5 COMPLETE!")
    print("=" * 60)
    print("\nOutputs:")
    print("  - emotion_arc.json        (emotional trajectory & phases)")
    if contextual_done:
        print("  - contextual_analysis.json (segment analysis with context)")
    print("  - refined_narrative.json  (unified coherent narrative)")


def run_pipeline(
    video_path: str = "scene.mp4",
    frames_dir: str = "frames",
    interval_seconds: float = 0.5,
    phase1: bool = True,
    phase2: bool = True,
    phase2_5: bool = True,
    max_frames: int = 10,
    skip_claude: bool = False,
    use_finetuned: bool = False
):

    if phase1:
        run_phase1(video_path, frames_dir, interval_seconds)
    
    if phase2:
        run_phase2(max_frames, skip_claude, use_finetuned)
    
    if phase2_5:
        run_phase2_5(max_frames)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Movie Scene Analyzer Pipeline")
    parser.add_argument("--video", default="scene.mp4", help="Input video path")
    parser.add_argument("--frames-dir", default="frames", help="Frames output directory")
    parser.add_argument("--interval", type=float, default=0.5, help="Frame extraction interval (seconds)")
    parser.add_argument("--phase1-only", action="store_true", help="Run only Phase 1")
    parser.add_argument("--phase2-only", action="store_true", help="Run only Phase 2")
    parser.add_argument("--phase2-5-only", action="store_true", help="Run only Phase 2.5 (contextual grounding)")
    parser.add_argument("--skip-phase2-5", action="store_true", help="Skip Phase 2.5")
    parser.add_argument("--max-frames", type=int, default=10, help="Max frames to send to LLMs")
    parser.add_argument("--skip-claude", action="store_true", help="Skip Claude analysis (GPT-4o only)")
    parser.add_argument("--use-finetuned", action="store_true", help="Use fine-tuned models if available")
    parser.add_argument("--finetune-status", action="store_true", help="Check fine-tuned model status and exit")
    
    args = parser.parse_args()
    
    # Just check fine-tune status if requested
    if args.finetune_status:
        if HAS_FINETUNE:
            print_model_status()
        else:
            print("Fine-tune module not available. Check finetune/ directory.")
        exit(0)
    
    # Determine which phases to run
    if args.phase2_5_only:
        phase1 = False
        phase2 = False
        phase2_5 = True
    else:
        phase1 = not args.phase2_only
        phase2 = not args.phase1_only
        phase2_5 = not args.skip_phase2_5 and not args.phase1_only and not args.phase2_only
    
    run_pipeline(
        video_path=args.video,
        frames_dir=args.frames_dir,
        interval_seconds=args.interval,
        phase1=phase1,
        phase2=phase2,
        phase2_5=phase2_5,
        max_frames=args.max_frames,
        skip_claude=args.skip_claude,
        use_finetuned=args.use_finetuned
    )
