# Project Title: Intelligent Movie Scene Analyzer and Explainer

## Objective:

Develop a web UI that ingests a 5-minute movie clip and produces an explanation including:

- Scene description (setting, location, time of day)
- Mood and cinematic style
- Actor emotions and tones per line
- Script annotation with emotion metadata
- Optional insights (themes, pacing, sound/score analysis)


## System Overview

1. Media Parsing
    - Video → frames, subtitles, audio

2. Multimodal Understanding
    - Visual scene/context understanding
    - Audio emotion and actor tonality analysis

3. Script/Transcript Annotation
    - Link dialogue to emotions

4. Semantic Scene Explanation Generation
    - Narrative paragraph or structured output

5. UI Presentation Layer
    - Visual timeline + annotated script
    - Emotion overlay, context panel


## LLM & Multimodal Model Selection

### 1. Video Frame Understanding Models

#### OpenAI's GPT-4o Vision / Vision Pro

Use For:
- Scene setting detection
- Object recognition (location, props, costumes)
- Cinematic style discussion

Why?
- Excellent at vision + reasoning
- Can provide textual descriptions grounded in visual context

Input: Sample frames extracted from the video
Output: Setting, objects, props, possible time of day


### 2. Audio & Voice Emotion Recognition

#### Faster Whisper (IMPLEMENTED)
- Model:`faster_whisper` with `medium` model size
- Use For: High-quality transcription with accurate timestamps
- Output:`transcript.json` with start/end times per segment

####  Wav2Vec2 Speech Emotion Recognition (IMPLEMENTED)
- Model:`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
- Framework: HuggingFace Transformers
- Use For: Emotion labeling per transcript segment
- Emotions: angry, calm, disgust, fear, happy, neutral, sad, surprise
- Output: `emotions.json` with emotion + confidence scores

---

### 3. Facial Expression & Actor Emotion Recognition

#### Vision Transformers (ViT / CLIP-based) (Planned)

Use For:
- Detect actor emotions frame-by-frame
- Lip sync / facial movement analysis

Why:
- CLIP embeddings enable semantic alignment between face and textual emotion descriptions.

**Implementation Considerations (per supervisor feedback):**
- With 0.5-second frame intervals, expect ~600 frames for a 5-minute clip (high volume)
- Use **DeepFace** as a baseline comparison to validate CLIP embeddings
- DeepFace integrates multiple facial recognition models for cross-validation


### 4. Narrative & Reasoning LLM

#### OpenAI GPT-4o / GPT-4o Pro

Use For:
- Generate narrative explanations
- Summarize script context
- Integrate visual + audio data

Why?
- Best general-purpose reasoning with multimodal grounding
- Can handle complex prompts

#### Anthropic Claude 3 / Claude 3.5 Sonnet

Use For:
- Alternate reasoning
- Cross-check narrative quality
- Safety filters

Why?
- Claude often offers more *coherent long-form generation*
- Helpful as a *secondary* model for comparison



## Integration Plan

### Phase 1 — Core Pipeline IN PROGRESS

Goal: Ingest video + transcript → emotional script annotations

| Step | Task | Status | Technology |
|------|------|--------|------------|
| 1.1 | Extract frames every 0.5 seconds | DONE | `ffmpeg` via `frames.py` |
| 1.2 | Generate transcript with timestamps | DONE | `faster_whisper` via `transcript.py` |
| 1.3 | Run speech emotion classifier | DONE | `wav2vec2-lg-xlsr-en-speech-emotion-recognition` via `emotion.py` |
| 1.4 | Pair frames + emotions to transcript | DONE | `pair.py` |
| 1.5 | Run facial emotion detector on frames | DONE | DeepFace via `face_emotion.py` |

Current Outputs:
- `transcript.json` — timestamped transcript segments
- `frames.json` — frame metadata with timestamps  
- `frames/` — extracted JPG frames
- `emotions.json` — speech emotion per segment
- `paired_data.json` — combined: frames + emotions + text

Pipeline Script: `pipeline.py` orchestrates steps 1.1–1.4


### Phase 2 — Semantic Understanding COMPLETE ✓

Goal: Generate textual explanations with cross-validation

| Step | Task | Status | Technology |
|------|------|--------|------------|
| 2.1 | Extract keyframes based on emotion significance | DONE | `keyframe_extractor.py` |
| 2.2 | Generate scene analysis with GPT-4o Vision | DONE | `scene_analyzer.py` |
| 2.3 | Cross-validate with Claude 3.5 Sonnet | DONE | `claude_analyzer.py` |
| 2.4 | Compare and merge analyses | DONE | `compare_analyses.py` |

Deliverables:
- `keyframes.json` — selected emotionally significant frames
- `conflict_summary.json` — audio/visual emotion mismatches
- `scene_analysis.json` — GPT-4o narrative analysis
- `claude_analysis.json` — Claude narrative analysis
- `analysis_comparison.json` — cross-validation with consensus analysis

**Implementation Considerations (per supervisor feedback):**

**Context Window Management:**
- A 5-minute clip at 0.5-second intervals produces ~600 frames
- Sending all frames to GPT-4o Vision may exceed token limits or become cost-prohibitive
- Implement **Keyframe Extraction** logic: select only the most "emotionally active" frames based on emotion spikes in `emotions.json`

**Multimodal Conflict Resolution:**
- A key research aspect: handling discrepancies between audio and visual signals
- Example: audio classified as "calm" but visual frame shows "anger" (common in sarcasm)
- Add a **"Conflict Resolution" prompt** for GPT-4o to interpret these multimodal conflicts
- This will add significant depth to the analyzer's narrative capabilities


### Phase 2.5 — Contextual Grounding NEW ✓

Goal: Improve narrative consistency and extract emotional arc intact

| Step | Task | Status | Technology |
|------|------|--------|------------|
| 2.5.1 | Extract emotional arc (phases, turning points) | DONE | `emotion_arc.py` |
| 2.5.2 | Contextual analysis with sliding window | DONE | `contextual_analyzer.py` |
| 2.5.3 | Narrative refinement and synthesis | DONE | `narrative_refiner.py` |

Deliverables:
- `emotion_arc.json` — emotional trajectory with phases (exposition, rising action, climax, resolution)
- `contextual_analysis.json` — segment analysis with sliding context window
- `refined_narrative.json` — unified coherent narrative synthesis

**Key Features (addressing supervisor feedback):**

**Contextual Grounding:**
- Sliding context window provides previous segment context to GPT-4o
- Bridges gap between "what is said" (dialogue) and "what is happening" (visual/emotional)
- Each segment analysis references previous context for continuity

**Emotional Arc Extraction:**
- Identifies narrative phases: exposition → rising action → climax → falling action → resolution
- Detects emotional turning points with significance scores
- Tracks valence (positive/negative) and intensity throughout the clip

**Narrative Consistency:**
- Final refinement pass synthesizes all analysis sources
- Resolves contradictions between GPT-4o and Claude
- Ensures unified narrative flow across all phases


### Phase 3 — UI / Output Visualization COMPLETE ✓

Goal: Build interactive timeline UI

| Step | Task | Status | Technology |
|------|------|--------|------------|
| 3.1 | Set up React + TypeScript project | DONE | Vite + React |
| 3.2 | Create video player with sync | DONE | HTML5 Video API |
| 3.3 | Build emotion timeline graph | DONE | Recharts |
| 3.4 | Add synchronized transcript panel | DONE | React |
| 3.5 | Create analysis panel with sections | DONE | React |

Features Implemented:
- ✅ Video + script synchronization (auto-scroll to current segment)
- ✅ Real-time emotion overlay (pulsing indicator synced to video)
- ✅ Stacked area chart showing emotion distribution over time
- ✅ Clickable transcript segments for seeking
- ✅ Expandable analysis sections (scene overview, characters, themes, etc.)
- ✅ Conflict indicators highlighting audio/visual mismatches
- ✅ Dark cinema-inspired theme

Tech Stack:
- React 19 + TypeScript
- Vite for build tooling
- Recharts for emotion visualization
- Lucide React for icons


## Current Project State

| Component | Technology | File |
|-----------|------------|------|
| Transcription | `faster-whisper` (Whisper medium) | `transcript.py` |
| Frame Extraction | `ffmpeg` | `frames.py` |
| Speech Emotion | `transformers` + `wav2vec2` | `emotion.py` |
| Facial Emotion | DeepFace | `face_emotion.py` |
| Data Pairing | Custom Python | `pair.py` |
| Keyframe Extraction | Custom Python | `keyframe_extractor.py` |
| Scene Analysis (GPT-4o) | OpenAI GPT-4o Vision | `scene_analyzer.py` |
| Scene Analysis (Claude) | Anthropic Claude 3.5 Sonnet | `claude_analyzer.py` |
| Cross-Validation | Custom Python | `compare_analyses.py` |
| **Emotion Arc** | Custom Python | `emotion_arc.py` |
| **Contextual Analysis** | OpenAI GPT-4o + sliding window | `contextual_analyzer.py` |
| **Narrative Refiner** | OpenAI GPT-4o | `narrative_refiner.py` |
| **Fine-Tuning** | OpenAI API + HuggingFace | `finetune/` |
| **Evaluation** | Custom Python | `evaluation/` |
| Orchestration | Python | `pipeline.py` |
| **Web UI** | React + TypeScript + Recharts | `ui/` |
| **API Server** | FastAPI + Uvicorn | `api.py` |
| Environment | Python 3.13 + venv | `venv/` |

**Critical Deliverable:** `paired_data.json`
- Bridges the gap between raw media and semantic meaning
- **Important:** Ensure timestamps are normalized to align with standard HTML5 video player timecodes for the final UI


## Dependencies

```
faster-whisper
transformers
torch
torchaudio
deepface
openai
anthropic
fastapi
uvicorn
python-multipart
```

## Running the Pipeline

```bash
# Activate virtual environment
source venv/bin/activate

# Run full pipeline (Phase 1 + Phase 2 + Phase 2.5)
python pipeline.py

# Run only Phase 1 (media parsing + emotion extraction)
python pipeline.py --phase1-only

# Run only Phase 2 with full cross-validation (needs both API keys)
export OPENAI_API_KEY='your-key-here'
export ANTHROPIC_API_KEY='your-key-here'
python pipeline.py --phase2-only

# Run Phase 2 with GPT-4o only (skip Claude)
export OPENAI_API_KEY='your-key-here'
python pipeline.py --phase2-only --skip-claude

# Run only Phase 2.5 (contextual grounding - requires Phase 2 outputs)
python pipeline.py --phase2-5-only

# Run Phase 1 + 2 without Phase 2.5
python pipeline.py --skip-phase2-5

# Customize frame extraction interval
python pipeline.py --interval 1.0

# Limit frames sent to LLMs (cost control)
python pipeline.py --phase2-only --max-frames 5
```

### Running the Web UI (with Video Upload)

The UI now supports uploading videos directly and running the full pipeline.

**Terminal 1 - Start the API server:**
```bash
source venv/bin/activate
export OPENAI_API_KEY='your-key-here'  # Optional, for AI scene analysis
python api.py
```

**Terminal 2 - Start the UI:**
```bash
cd ui
npm run dev
```

Then open `http://localhost:5173` and upload a video (max 5 minutes).

### Running the Web UI (Static Mode - Pre-analyzed Data)

If you've already run the pipeline and want to view existing results:

```bash
cd ui

# Copy data files
cp ../paired_data.json public/data/
cp ../scene_analysis.json public/data/
cp ../scene.mp4 public/data/

npm run dev
```

## Next Steps

1. ~~**Phase 1:** Core pipeline with facial emotion detection~~ ✅ DONE
2. ~~**Phase 2:** Semantic understanding with GPT-4o + Claude cross-validation~~ ✅ DONE
3. ~~**Phase 2.5:** Contextual grounding and narrative consistency~~ ✅ DONE
4. ~~**Phase 3:** React UI with emotion timeline and synchronized transcript~~ ✅ DONE

### Evaluation Module NEW

Goal: Quantify system performance with metrics

```bash
# Step 1: Create ground truth labels (interactive CLI)
python -m evaluation.labeling_tool

# Step 2: Run all evaluations
python -m evaluation.run_evaluation

# Optional: Include LLM-as-judge
python -m evaluation.run_evaluation --llm-judge
```

| Metric | What It Measures | Target |
|--------|------------------|--------|
| Emotion Accuracy | Correct emotion predictions | >70% |
| Conflict Precision | Valid conflict detections | >80% |
| Narrative Coherence | LLM-as-judge rubric score (1-5) | >3.5/5 |

**LLM-as-Judge Dimensions:**
- Logical Flow, Temporal Consistency, Emotion Arc Coherence
- Visual/Audio Grounding, Conflict Interpretation
- Validated against small human-labeled subset for correlation

Outputs: `evaluation/results/full_evaluation.json`

See `evaluation/README.md` for detailed instructions.


### Phase 4 — Fine-Tuning (Optional) NEW

Goal: Fine-tune existing LLMs for improved domain-specific performance

| Script | Purpose | Output |
|--------|---------|--------|
| `finetune/create_training_data.py` | Generate training data from analyses | `training_data/*.jsonl` |
| `finetune/openai_finetune.py` | Fine-tune GPT-4o-mini on scene analysis | Fine-tuned model |
| `finetune/emotion_finetune.py` | Fine-tune wav2vec2 on movie emotions | `emotion_model/` |

```bash
# Generate training data from your existing analyses
python -m finetune.create_training_data --format openai

# Fine-tune GPT-4o-mini (~$0.80 for 100 examples)
python -m finetune.openai_finetune --train finetune/training_data/train.jsonl

# Fine-tune emotion classifier on MELD dataset
python -m finetune.emotion_finetune --dataset meld --use-lora

# Run pipeline with fine-tuned models
python pipeline.py --use-finetuned
```

See `finetune/README.md` for detailed instructions.


### Future Enhancements (Optional)
- Add CLIP embeddings for semantic frame search
- Implement speaker diarization
- Add export to WebVTT/SRT subtitle formats
- Deploy UI to cloud hosting
- Evaluation metrics dashboard (accuracy, coherence scores)
