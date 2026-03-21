"""
Fine-Tuning Module for Movie Scene Analyzer

This module provides tools to fine-tune models for:
1. Scene narrative generation (GPT-4o-mini)
2. Speech emotion classification (wav2vec2)

Quick Start:
    # Check what fine-tuned models are available
    from finetune import print_model_status
    print_model_status()
    
    # Use fine-tuned scene analysis
    from finetune import analyze_scene_finetuned
    result = analyze_scene_finetuned(keyframes, conflicts)
    
    # Use fine-tuned emotion classification
    from finetune import classify_emotion_finetuned
    emotion, confidence, scores = classify_emotion_finetuned("audio.wav")
"""

from .use_finetuned import (
    # Model status
    get_model_status,
    print_model_status,
    get_finetuned_gpt_model_id,
    get_finetuned_emotion_model_path,
    
    # Scene analysis
    analyze_scene_finetuned,
    
    # Emotion classification
    classify_emotion_finetuned,
    classify_emotions_batch,
    load_emotion_model
)

__all__ = [
    "get_model_status",
    "print_model_status",
    "get_finetuned_gpt_model_id",
    "get_finetuned_emotion_model_path",
    "analyze_scene_finetuned",
    "classify_emotion_finetuned",
    "classify_emotions_batch",
    "load_emotion_model"
]
