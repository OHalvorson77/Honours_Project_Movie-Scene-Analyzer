# Honours Project Presentation Outline
## Intelligent Movie Scene Analyzer and Explainer

**Total time: 15 minutes** (including ~1 min setup + ~11 min talk + ~3–4 min Q&A)

---

## SLIDE 1: Title (30 sec)
**Content:**
- **Intelligent Movie Scene Analyzer**
- Your name
- Date
- Sparse, clean

**Say:** "Today I'll show you a system that watches a movie clip and explains what's happening—including the emotions we might miss if we only listened or only looked."

---

## SLIDE 2: The Problem (1 min)
**Content:**
- **What we wanted**
  - Explain what happens in a movie scene
  - Not just "what" is said—but *how* and *why*
- **One example**
  - Character says something calmly… but their face shows anger

**Say:** "We want to explain movie scenes in a way that captures emotion and subtext. Sometimes the voice says one thing and the face says another—like sarcasm. That's hard to see with only one cue."

---

## SLIDE 3: Why It's Hard (1 min)
**Content:**
- **Single cues can mislead**
  - Voice alone: might sound calm
  - Face alone: might look angry
  - Together: sarcasm
- **Short phrase:** "We need both to get the full picture"

**Say:** "If we only listen, we miss facial expressions. If we only look, we miss tone of voice. Real emotion often needs both."

---

## SLIDE 4: Our Approach (1 min)
**Content:**
- **One sentence**
  - "We combine what we hear, what we see, and AI reasoning to produce scene explanations."
- **Simple flow**
  - Video → Extract & Analyze → Explanation

**Say:** "Our system takes a video, extracts what we hear and what we see, finds where they disagree, and uses AI to explain what that means."

---

## SLIDE 5: High-Level Overview (1 min)
**Content:**
- **Simple 3-step diagram**
  1. Video in
  2. Pipeline (frames + transcript + emotions)
  3. Explanation out (timeline, transcript, narrative)
- **No technical labels**

**Say:** "At a high level: video goes in, we process it in stages, and out comes a timeline with an annotated transcript and a narrative explanation."

---

## SLIDE 6: What the System Does (2 min)
**Content:**
- **Bullets (short)**
  - Extracts key frames from the video
  - Transcribes what's said
  - Detects emotion from voice and face
  - Finds mismatches (e.g. calm voice + angry face)
  - Generates a narrative explanation
- **No model names here**

**Say:** "We extract frames, transcribe speech, detect emotion from both voice and face, flag places where they disagree, and then use AI to write a coherent explanation of the scene."

---

## SLIDE 7: Example—A Real Conflict (2 min)
**Content:**
- **Quote:** "That's still on."
- **Data:** Voice = neutral | Face = fear
- **Interpretation:** "Suppressed fear—character hides worry behind a calm voice"
- **One short takeaway:** "Mismatches like this are exactly what we want to find"

**Say:** "In one clip, a character says 'That's still on' in a neutral tone, but their face shows fear. Our system labels this as suppressed fear and explains that the character is hiding worry. A single-modality system would miss that."

---

## SLIDE 8: Example Output (1 min)
**Content:**
- **Screenshot or mock** of the UI
  - Video player + transcript + emotion timeline
- **Caption:** "Interactive timeline with video, transcript, and emotion overlay"

**Say:** "The final output is an interactive UI: you can watch the video, see the transcript with emotions, and read the narrative analysis."

---

## SLIDE 9: Architecture (1.5 min) *(Technical—near end)*
**Content:**
- **Simple diagram**
  - Box 1: Data (frames, transcript, emotions)
  - Box 2: Fusion & conflict detection
  - Box 3: LLMs (GPT-4o, Claude) for narrative
- **3–4 labels max**

**Say:** "Architecturally, we first extract frames and transcript, then fuse emotions from voice and face and detect conflicts. Finally, we send curated keyframes and conflicts to GPT-4o and Claude to produce the narrative."

---

## SLIDE 10: Evaluation (1 min)
**Content:**
- **Three metrics**
  - Emotion accuracy (vs. human labels)
  - Conflict precision (are mismatches meaningful?)
  - Narrative quality (LLM judge)
- **Targets:** >70% accuracy, >80% conflict precision

**Say:** "We evaluate with human labels for emotions and conflicts, and use an LLM judge to score narrative quality. We aim for over 70% emotion accuracy and over 80% conflict precision."

---

## SLIDE 11: Limitations & Future Work (30 sec)
**Content:**
- **Bullets**
  - Low confidence on some speech segments
  - No speaker separation (who says what)
  - Possible extensions: better models, longer clips

**Say:** "Limitations: some speech segments have low confidence, and we don't separate speakers. Future work could improve emotion models and support longer clips."

---

## SLIDE 12: Summary & Thank You (30 sec)
**Content:**
- **What we built**
  - Multimodal pipeline for movie scene explanation
  - Conflict detection for sarcasm, masked emotion
  - Interactive UI with narrative analysis
- **Thank you / Questions**

**Say:** "To wrap up: we built a pipeline that combines voice and face to explain movie scenes, detects mismatches like sarcasm and suppressed emotion, and presents results in an interactive UI. Thank you—I'm happy to take questions."

---

## DEMO OPTION
If you do a live demo, use it to replace or follow Slide 8.
- Upload a short clip (e.g. 1 min)
- Show the UI: video, transcript, emotion timeline
- Point out 1–2 conflict segments
- Keep it under 2–3 minutes

---

## SPEAKER NOTES
- **Don’t read slides.** Use bullets as prompts; expand in your own words.
- **First 5 min:** Slides 1–5. Keep it conceptual and simple.
- **Middle:** Slides 6–8. Concrete examples and output.
- **Technical:** Slides 9–10. Only if time allows or for technical audiences.
- **Practice:** Rehearse 2–3 times to hit ~11 minutes.
- **Q&A prep:** Be ready to explain keyframe extraction, conflict types, and evaluation in one sentence each.
