"""
Emotion Classifier Fine-Tuning for Movie Dialogue

Fine-tunes the wav2vec2 speech emotion model on movie-specific data.
This addresses the professor's suggestion to fine-tune existing models
rather than just using them as-is.

The idea is to improve emotion classification accuracy on movie dialogue,
which often contains:
- Acted emotions (different from natural speech)
- Sarcasm and irony
- Emotional complexity

Datasets to consider:
- MELD (Multimodal EmotionLines Dataset) - TV show dialogues
- IEMOCAP - Acted emotional speech
- Your own labeled data from movie clips

Prerequisites:
    pip install transformers datasets peft accelerate torchaudio

Usage:
    # Using MELD dataset (TV show emotions)
    python emotion_finetune.py --dataset meld --epochs 3
    
    # Using custom labeled data
    python emotion_finetune.py --custom-data ./labeled_clips/
    
    # With LoRA for efficient fine-tuning
    python emotion_finetune.py --dataset meld --use-lora
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings("ignore")

# Check for required packages
try:
    import torch
    import torchaudio
    from transformers import (
        Wav2Vec2ForSequenceClassification,
        Wav2Vec2FeatureExtractor,
        TrainingArguments,
        Trainer
    )
    from datasets import Dataset, load_dataset, Audio
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Missing packages. Install with:")
    print("  pip install transformers datasets torchaudio accelerate")

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


# Emotion labels (matching wav2vec2-lg-xlsr-en-speech-emotion-recognition)
EMOTION_LABELS = ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABEL2ID = {label: i for i, label in enumerate(EMOTION_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(EMOTION_LABELS)}

# MELD emotion mapping (MELD uses different labels)
MELD_EMOTION_MAP = {
    "anger": "angry",
    "disgust": "disgust", 
    "fear": "fear",
    "joy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprise"
}


def load_meld_dataset(split: str = "train") -> Dataset:
    """
    Load MELD dataset (Multimodal EmotionLines Dataset).
    Contains emotion-labeled clips from the TV show Friends.
    """
    print(f"Loading MELD dataset ({split})...")
    
    # MELD is available on Hugging Face
    dataset = load_dataset("declare-lab/MELD", split=split, trust_remote_code=True)
    
    # Filter to only audio-available samples
    # Note: MELD provides text, audio, and video - we use audio
    
    return dataset


def create_custom_dataset(data_dir: str) -> Dataset:
    """
    Create dataset from custom labeled audio files.
    
    Expected directory structure:
        data_dir/
            angry/
                clip1.wav
                clip2.wav
            happy/
                clip1.wav
            ...
    
    Or a metadata.json file:
        [
            {"path": "clip1.wav", "emotion": "angry"},
            {"path": "clip2.wav", "emotion": "happy"},
            ...
        ]
    """
    data_path = Path(data_dir)
    
    # Check for metadata.json
    metadata_path = data_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Load audio files
        data = []
        for item in metadata:
            audio_path = data_path / item["path"]
            if audio_path.exists():
                data.append({
                    "audio": str(audio_path),
                    "emotion": item["emotion"],
                    "label": LABEL2ID.get(item["emotion"], LABEL2ID["neutral"])
                })
        
        return Dataset.from_dict({
            "audio": [d["audio"] for d in data],
            "emotion": [d["emotion"] for d in data],
            "label": [d["label"] for d in data]
        })
    
    # Otherwise, use directory structure
    data = []
    for emotion in EMOTION_LABELS:
        emotion_dir = data_path / emotion
        if emotion_dir.exists():
            for audio_file in emotion_dir.glob("*.wav"):
                data.append({
                    "audio": str(audio_file),
                    "emotion": emotion,
                    "label": LABEL2ID[emotion]
                })
            for audio_file in emotion_dir.glob("*.mp3"):
                data.append({
                    "audio": str(audio_file),
                    "emotion": emotion,
                    "label": LABEL2ID[emotion]
                })
    
    if not data:
        raise ValueError(f"No audio files found in {data_dir}")
    
    print(f"Found {len(data)} audio files")
    
    return Dataset.from_dict({
        "audio": [d["audio"] for d in data],
        "emotion": [d["emotion"] for d in data],
        "label": [d["label"] for d in data]
    })


def preprocess_meld_example(example: Dict, feature_extractor: Wav2Vec2FeatureExtractor) -> Dict:
    """Preprocess a MELD dataset example."""
    # Map MELD emotion to our labels
    meld_emotion = example.get("Emotion", "neutral").lower()
    emotion = MELD_EMOTION_MAP.get(meld_emotion, "neutral")
    
    # Load and process audio
    audio = example.get("audio")
    if audio is not None:
        # Resample to 16kHz if needed
        speech_array = audio["array"]
        sampling_rate = audio["sampling_rate"]
        
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            speech_array = resampler(torch.tensor(speech_array)).numpy()
        
        inputs = feature_extractor(
            speech_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            max_length=16000 * 10,  # 10 seconds max
            truncation=True
        )
        
        return {
            "input_values": inputs.input_values.squeeze(),
            "label": LABEL2ID.get(emotion, LABEL2ID["neutral"])
        }
    
    return None


def preprocess_custom_example(example: Dict, feature_extractor: Wav2Vec2FeatureExtractor) -> Dict:
    """Preprocess a custom dataset example."""
    audio_path = example["audio"]
    
    # Load audio
    speech_array, sampling_rate = torchaudio.load(audio_path)
    speech_array = speech_array.squeeze().numpy()
    
    # Resample to 16kHz if needed
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        speech_array = resampler(torch.tensor(speech_array)).numpy()
    
    # Extract features
    inputs = feature_extractor(
        speech_array,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        max_length=16000 * 10,  # 10 seconds max
        truncation=True
    )
    
    return {
        "input_values": inputs.input_values.squeeze(),
        "label": example["label"]
    }


def load_model_and_extractor(model_name: str, use_lora: bool = False):
    """Load the pretrained model and feature extractor."""
    print(f"Loading model: {model_name}")
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(EMOTION_LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL
    )
    
    if use_lora and HAS_PEFT:
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    elif use_lora and not HAS_PEFT:
        print("Warning: peft not installed, training full model")
        print("Install with: pip install peft")
    
    return model, feature_extractor


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute evaluation metrics."""
    import numpy as np
    
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    labels = eval_pred.label_ids
    
    accuracy = (predictions == labels).mean()
    
    # Per-class accuracy
    per_class_acc = {}
    for i, emotion in enumerate(EMOTION_LABELS):
        mask = labels == i
        if mask.sum() > 0:
            per_class_acc[f"acc_{emotion}"] = (predictions[mask] == labels[mask]).mean()
    
    return {"accuracy": accuracy, **per_class_acc}


def train(
    model,
    feature_extractor,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    output_dir: str = "./emotion_model",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5
):
    """Train the emotion classifier."""
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="accuracy" if eval_dataset else None,
        push_to_hub=False,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=4,
        report_to="none"  # Disable wandb/tensorboard
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics if eval_dataset else None
    )
    
    print("\nStarting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model(output_dir)
    feature_extractor.save_pretrained(output_dir)
    
    print(f"\nModel saved to {output_dir}")
    
    return trainer


def create_training_data_from_paired(
    paired_data_path: str = "paired_data.json",
    audio_dir: str = "audio_segments",
    output_path: str = "finetune/emotion_data/metadata.json"
) -> List[Dict]:
    """
    Create labeled training data from your paired_data.json.
    
    This requires you to have extracted audio segments from your video.
    You can use ffmpeg to extract segments:
        ffmpeg -i scene.mp4 -ss 0.0 -to 2.5 -vn audio_segments/seg_0.wav
    """
    with open(paired_data_path, "r") as f:
        paired_data = json.load(f)
    
    training_data = []
    for i, seg in enumerate(paired_data):
        emotion = seg.get("speech_emotion") or seg.get("fused_emotion")
        if emotion and emotion in EMOTION_LABELS:
            audio_path = f"{audio_dir}/seg_{i}.wav"
            if Path(audio_path).exists():
                training_data.append({
                    "path": f"seg_{i}.wav",
                    "emotion": emotion,
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg.get("text"),
                    "confidence": seg.get("speech_confidence")
                })
    
    # Save metadata
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Created {len(training_data)} training examples")
    print(f"Saved to {output_path}")
    
    return training_data


def extract_audio_segments(
    video_path: str = "scene.mp4",
    paired_data_path: str = "paired_data.json",
    output_dir: str = "finetune/emotion_data"
):
    """
    Extract audio segments from video based on paired_data timestamps.
    Requires ffmpeg.
    """
    import subprocess
    
    with open(paired_data_path, "r") as f:
        paired_data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Extracting {len(paired_data)} audio segments...")
    
    for i, seg in enumerate(paired_data):
        start = seg.get("start", 0)
        end = seg.get("end", start + 1)
        duration = end - start
        
        output_path = os.path.join(output_dir, f"seg_{i}.wav")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(start),
            "-t", str(duration),
            "-vn",  # No video
            "-acodec", "pcm_s16le",
            "-ar", "16000",  # 16kHz for wav2vec2
            "-ac", "1",  # Mono
            output_path
        ]
        
        subprocess.run(cmd, capture_output=True)
    
    print(f"Extracted segments to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune emotion classifier for movie dialogue")
    
    # Data source
    parser.add_argument("--dataset", choices=["meld", "custom"], 
                       help="Dataset to use (meld or custom)")
    parser.add_argument("--custom-data", metavar="DIR",
                       help="Directory with custom labeled audio")
    parser.add_argument("--extract-segments", action="store_true",
                       help="Extract audio segments from video first")
    
    # Model options
    parser.add_argument("--model", 
                       default="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                       help="Base model to fine-tune")
    parser.add_argument("--use-lora", action="store_true",
                       help="Use LoRA for efficient fine-tuning")
    
    # Training options
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--output-dir", default="./finetune/emotion_model",
                       help="Output directory for fine-tuned model")
    
    args = parser.parse_args()
    
    if not HAS_TRANSFORMERS:
        print("Required packages not installed. Exiting.")
        return
    
    # Step 1: Extract audio segments if requested
    if args.extract_segments:
        print("=" * 60)
        print("STEP 1: Extracting audio segments")
        print("=" * 60)
        extract_audio_segments()
        create_training_data_from_paired()
        print("\nAudio segments extracted. Now run again with --custom-data")
        return
    
    # Step 2: Load model
    print("=" * 60)
    print("Loading model and feature extractor")
    print("=" * 60)
    model, feature_extractor = load_model_and_extractor(args.model, args.use_lora)
    
    # Step 3: Load dataset
    print("\n" + "=" * 60)
    print("Loading dataset")
    print("=" * 60)
    
    if args.dataset == "meld":
        train_dataset = load_meld_dataset("train")
        eval_dataset = load_meld_dataset("validation")
        
        # Preprocess
        print("Preprocessing MELD dataset...")
        train_dataset = train_dataset.map(
            lambda x: preprocess_meld_example(x, feature_extractor),
            remove_columns=train_dataset.column_names
        ).filter(lambda x: x is not None)
        
        eval_dataset = eval_dataset.map(
            lambda x: preprocess_meld_example(x, feature_extractor),
            remove_columns=eval_dataset.column_names
        ).filter(lambda x: x is not None)
        
    elif args.custom_data:
        dataset = create_custom_dataset(args.custom_data)
        
        # Preprocess
        print("Preprocessing custom dataset...")
        dataset = dataset.map(
            lambda x: preprocess_custom_example(x, feature_extractor)
        )
        
        # Split into train/eval
        split = dataset.train_test_split(test_size=0.2)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        print("Please specify --dataset meld or --custom-data <dir>")
        return
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")
    
    # Step 4: Train
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    
    trainer = train(
        model=model,
        feature_extractor=feature_extractor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    # Step 5: Evaluate
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    results = trainer.evaluate()
    print(f"\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save model info
    model_info = {
        "base_model": args.model,
        "fine_tuned_on": args.dataset or "custom",
        "use_lora": args.use_lora,
        "epochs": args.epochs,
        "final_accuracy": results.get("eval_accuracy", 0),
        "output_dir": args.output_dir
    }
    
    with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\n✅ Fine-tuning complete!")
    print(f"Model saved to: {args.output_dir}")
    print(f"\nTo use in your pipeline, update emotion.py to load from:")
    print(f"  model_path = '{args.output_dir}'")


if __name__ == "__main__":
    main()
