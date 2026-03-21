# Evaluation Module

Tools to evaluate the Movie Scene Analyzer's performance with quantitative metrics.

## Quick Start

### Step 1: Create Ground Truth Labels

```bash
# Interactive labeling tool
python -m evaluation.labeling_tool

# Or run specific modes:
python -m evaluation.labeling_tool --mode emotion    # Label emotions
python -m evaluation.labeling_tool --mode conflict   # Validate conflicts
python -m evaluation.labeling_tool --mode narrative  # Rate narrative quality

# Check progress
python -m evaluation.labeling_tool --status
```

### Step 2: Run Evaluation

```bash
# Run all evaluations
python -m evaluation.run_evaluation

# Include LLM-as-judge (requires ANTHROPIC_API_KEY)
python -m evaluation.run_evaluation --llm-judge

# Generate text report
python -m evaluation.run_evaluation --report
```

## Evaluation Metrics

### 1. Emotion Classification Accuracy

Compares model-predicted emotions against your manual labels.

| Metric | Description | Target |
|--------|-------------|--------|
| Accuracy | % of correct predictions | >70% |
| F1 Score | Harmonic mean of precision/recall | >0.6 |
| Per-class | Breakdown by emotion type | Varies |

**Output:** `results/emotion_evaluation.json`

### 2. Conflict Detection Precision

Validates whether detected audio/visual conflicts are meaningful.

| Metric | Description | Target |
|--------|-------------|--------|
| Precision | % of valid conflicts | >80% |
| By Type | Breakdown by conflict type | Varies |

**Output:** `results/conflict_evaluation.json`

### 3. Narrative Quality (LLM-as-Judge)

Automated evaluation using GPT-4 with a detailed rubric scoring:

| Dimension | Description | Weight |
|-----------|-------------|--------|
| Logical Flow | Does the narrative progress logically? | 20% |
| Temporal Consistency | Events in correct chronological order? | 15% |
| Emotion Arc Coherence | Does emotional journey align with data? | 25% |
| Visual/Audio Grounding | References specific cues appropriately? | 20% |
| Conflict Interpretation | Are conflicts interpreted meaningfully? | 10% |
| Overall Coherence | Holistic quality assessment | 10% |

```bash
# Run LLM-as-judge evaluation
python -m evaluation.llm_judge

# Use Claude instead of GPT-4
python -m evaluation.llm_judge --model claude

# Validate LLM scores against human labels
python -m evaluation.llm_judge --validate
```

**Output:** `results/llm_judge_evaluation.json`

**Validation:** Compare LLM judge scores with a small human-evaluated subset (~10-15 scenes) to ensure correlation.

## Recommended Labeling Targets

For statistically meaningful results:

| Type | Minimum | Recommended |
|------|---------|-------------|
| Emotion labels | 30 | 50+ |
| Conflict validations | All detected | All |
| Narrative rating | 1 | 3+ (different clips) |

## Files

```
evaluation/
├── __init__.py
├── README.md
├── labeling_tool.py      # Create ground truth labels
├── evaluate_emotions.py  # Emotion accuracy metrics
├── evaluate_conflicts.py # Conflict precision metrics
├── evaluate_narrative.py # Narrative quality metrics
├── run_evaluation.py     # Run all + generate report
├── labels/               # Your ground truth labels
│   ├── emotion_labels.json
│   ├── conflict_labels.json
│   └── narrative_labels.json
└── results/              # Evaluation results
    ├── emotion_evaluation.json
    ├── conflict_evaluation.json
    ├── narrative_evaluation.json
    ├── full_evaluation.json
    └── evaluation_report.txt
```

## Using Results for Your Report

After running evaluation, you can include metrics like:

> "The emotion classification system achieved 72% accuracy (F1=0.68) on 50 
> manually labeled segments. Conflict detection precision was 85%, with 
> sarcasm/irony cases showing the highest validity rate (92%). Narrative 
> quality was rated 3.8/5 by human evaluation."

## Tips

1. **Be consistent**: Use the same criteria when labeling all segments
2. **Include edge cases**: Don't skip difficult examples
3. **Document disagreements**: Note when you're uncertain
4. **Re-evaluate after changes**: Run evaluation after fine-tuning to measure improvement
