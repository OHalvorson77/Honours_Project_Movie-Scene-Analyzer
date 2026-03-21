"""
Inference with Fine-Tuned Models

This module provides functions to use your fine-tuned models:
1. Fine-tuned GPT-4o-mini for scene analysis
2. Fine-tuned emotion classifier for speech emotion

Usage:
    from finetune.use_finetuned import (
        analyze_scene_finetuned,
        classify_emotion_finetuned,
        get_finetuned_model_id
    )
    
    # Scene analysis with fine-tuned GPT
    analysis = analyze_scene_finetuned(keyframes, conflict_summary)
    
    # Emotion classification with fine-tuned model
    emotion, confidence = classify_emotion_finetuned(audio_path)
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# OpenAI imports
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Transformers imports for emotion model
try:
    import torch
    import torchaudio
    from transformers import (
        Wav2Vec2ForSequenceClassification,
        Wav2Vec2FeatureExtractor
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# Model info paths
FINETUNE_DIR = Path(__file__).parent
GPT_MODEL_INFO = FINETUNE_DIR / "model_info.json"
EMOTION_MODEL_DIR = FINETUNE_DIR / "emotion_model"

# Emotion labels
EMOTION_LABELS = ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def get_finetuned_gpt_model_id() -> Optional[str]:
    """
    Get the fine-tuned GPT model ID from model_info.json.
    Returns None if no fine-tuned model is available.
    """
    if GPT_MODEL_INFO.exists():
        with open(GPT_MODEL_INFO, "r") as f:
            info = json.load(f)
            return info.get("model_id")
    return None


def get_finetuned_emotion_model_path() -> Optional[str]:
    """
    Get the path to the fine-tuned emotion model.
    Returns None if no fine-tuned model is available.
    """
    if EMOTION_MODEL_DIR.exists() and (EMOTION_MODEL_DIR / "config.json").exists():
        return str(EMOTION_MODEL_DIR)
    return None


# ============================================================================
# Scene Analysis with Fine-Tuned GPT
# ============================================================================

def build_scene_prompt(
    keyframes: List[Dict[str, Any]],
    conflict_summary: Dict[str, Any]
) -> str:
    """Build prompt for scene analysis (same format as training data)."""
    lines = ["Analyze this movie scene based on the following transcript and emotion data:\n"]
    
    for kf in keyframes:
        emotion = kf.get("fused_emotion") or kf.get("speech_emotion") or "unknown"
        conflict = ""
        if kf.get("conflict_type"):
            conflict = f" [CONFLICT: voice={kf.get('speech_emotion')}, face={kf.get('face_emotion')}]"
        
        lines.append(f"[{kf['start']:.1f}s - {kf.get('end', kf['start'] + 1):.1f}s] \"{kf.get('text', '')}\"\n"
                    f"  Emotion: {emotion}{conflict}")
    
    if conflict_summary.get("conflicts"):
        lines.append("\n\nEmotion conflicts detected:")
        for c in conflict_summary["conflicts"][:3]:
            lines.append(f"  - {c['time']}: voice={c['speech_emotion']}, face={c['face_emotion']}")
    
    lines.append("\nProvide a comprehensive scene analysis including:")
    lines.append("1. Scene overview (setting, mood, atmosphere)")
    lines.append("2. Character emotional states and arcs")
    lines.append("3. Thematic observations")
    lines.append("4. Interpretation of any audio/visual emotion conflicts")
    
    return "\n".join(lines)


def analyze_scene_finetuned(
    keyframes: List[Dict[str, Any]],
    conflict_summary: Dict[str, Any],
    model_id: Optional[str] = None,
    fallback_to_base: bool = True
) -> Dict[str, Any]:
    """
    Analyze a scene using the fine-tuned GPT model.
    
    Args:
        keyframes: List of keyframe objects
        conflict_summary: Conflict summary dict
        model_id: Fine-tuned model ID (auto-detected if None)
        fallback_to_base: If True, use gpt-4o-mini if no fine-tuned model
    
    Returns:
        Analysis result dict
    """
    if not HAS_OPENAI:
        raise ImportError("openai package not installed")
    
    # Get model ID
    if model_id is None:
        model_id = get_finetuned_gpt_model_id()
    
    if model_id is None:
        if fallback_to_base:
            print("No fine-tuned model found, using gpt-4o-mini")
            model_id = "gpt-4o-mini"
        else:
            raise ValueError("No fine-tuned model available. Run openai_finetune.py first.")
    
    # Initialize client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Build prompt
    prompt = build_scene_prompt(keyframes, conflict_summary)
    
    print(f"Analyzing scene with model: {model_id}")
    
    # Call API
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": "You are an expert film analyst specializing in multimodal scene interpretation."
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.7
    )
    
    result = {
        "analysis": response.choices[0].message.content,
        "model": model_id,
        "is_finetuned": model_id != "gpt-4o-mini",
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens
        }
    }
    
    return result


# ============================================================================
# Emotion Classification with Fine-Tuned Model
# ============================================================================

# Global model cache
_emotion_model = None
_emotion_extractor = None


def load_emotion_model(model_path: Optional[str] = None):
    """Load the fine-tuned emotion model (cached)."""
    global _emotion_model, _emotion_extractor
    
    if _emotion_model is not None and _emotion_extractor is not None:
        return _emotion_model, _emotion_extractor
    
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers package not installed")
    
    # Get model path
    if model_path is None:
        model_path = get_finetuned_emotion_model_path()
    
    if model_path is None:
        # Fallback to original model
        print("No fine-tuned emotion model found, using original")
        model_path = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    
    print(f"Loading emotion model: {model_path}")
    
    _emotion_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    _emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        _emotion_model = _emotion_model.cuda()
    
    _emotion_model.eval()
    
    return _emotion_model, _emotion_extractor


def classify_emotion_finetuned(
    audio_path: str,
    model_path: Optional[str] = None
) -> Tuple[str, float, Dict[str, float]]:
    """
    Classify emotion from an audio file using the fine-tuned model.
    
    Args:
        audio_path: Path to audio file (wav, mp3)
        model_path: Path to fine-tuned model (auto-detected if None)
    
    Returns:
        Tuple of (emotion_label, confidence, all_scores)
    """
    model, extractor = load_emotion_model(model_path)
    
    # Load audio
    speech_array, sampling_rate = torchaudio.load(audio_path)
    speech_array = speech_array.squeeze()
    
    # Resample to 16kHz if needed
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        speech_array = resampler(speech_array)
    
    speech_array = speech_array.numpy()
    
    # Extract features
    inputs = extractor(
        speech_array,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    
    # Move to same device as model
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze()
    
    # Get results
    if torch.cuda.is_available():
        probs = probs.cpu()
    
    probs = probs.numpy()
    
    predicted_idx = probs.argmax()
    predicted_emotion = EMOTION_LABELS[predicted_idx]
    confidence = float(probs[predicted_idx])
    
    all_scores = {
        EMOTION_LABELS[i]: float(probs[i])
        for i in range(len(EMOTION_LABELS))
    }
    
    return predicted_emotion, confidence, all_scores


def classify_emotions_batch(
    audio_paths: List[str],
    model_path: Optional[str] = None
) -> List[Tuple[str, float, Dict[str, float]]]:
    """Classify emotions for multiple audio files."""
    return [classify_emotion_finetuned(path, model_path) for path in audio_paths]


# ============================================================================
# Integration with existing pipeline
# ============================================================================

def get_model_status() -> Dict[str, Any]:
    """
    Get status of fine-tuned models.
    Useful for checking what's available before running the pipeline.
    """
    status = {
        "gpt_model": {
            "available": False,
            "model_id": None,
            "info_path": str(GPT_MODEL_INFO)
        },
        "emotion_model": {
            "available": False,
            "model_path": None,
            "info_path": str(EMOTION_MODEL_DIR)
        }
    }
    
    # Check GPT model
    gpt_id = get_finetuned_gpt_model_id()
    if gpt_id:
        status["gpt_model"]["available"] = True
        status["gpt_model"]["model_id"] = gpt_id
    
    # Check emotion model
    emotion_path = get_finetuned_emotion_model_path()
    if emotion_path:
        status["emotion_model"]["available"] = True
        status["emotion_model"]["model_path"] = emotion_path
        
        # Load training info if available
        info_path = EMOTION_MODEL_DIR / "training_info.json"
        if info_path.exists():
            with open(info_path, "r") as f:
                status["emotion_model"]["training_info"] = json.load(f)
    
    return status


def print_model_status():
    """Print fine-tuned model status."""
    status = get_model_status()
    
    print("\n" + "=" * 60)
    print("FINE-TUNED MODEL STATUS")
    print("=" * 60)
    
    print("\n📝 GPT Scene Analysis Model:")
    if status["gpt_model"]["available"]:
        print(f"  ✅ Available: {status['gpt_model']['model_id']}")
    else:
        print(f"  ❌ Not available")
        print(f"     Run: python finetune/openai_finetune.py --train <data>")
    
    print("\n🎭 Emotion Classification Model:")
    if status["emotion_model"]["available"]:
        print(f"  ✅ Available: {status['emotion_model']['model_path']}")
        if "training_info" in status["emotion_model"]:
            info = status["emotion_model"]["training_info"]
            print(f"     Accuracy: {info.get('final_accuracy', 'N/A'):.2%}")
            print(f"     Trained on: {info.get('fine_tuned_on', 'N/A')}")
    else:
        print(f"  ❌ Not available")
        print(f"     Run: python finetune/emotion_finetune.py --dataset meld")
    
    print()


if __name__ == "__main__":
    print_model_status()
