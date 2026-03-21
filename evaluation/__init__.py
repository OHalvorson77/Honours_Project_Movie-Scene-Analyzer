"""
Evaluation Module for Movie Scene Analyzer

Provides tools to evaluate system performance:
1. Emotion classification accuracy
2. Conflict detection precision
3. Narrative quality scores

Quick Start:
    # Step 1: Create ground truth labels
    python -m evaluation.labeling_tool
    
    # Step 2: Run evaluation
    python -m evaluation.run_evaluation
"""

from pathlib import Path

EVALUATION_DIR = Path(__file__).parent
LABELS_DIR = EVALUATION_DIR / "labels"
RESULTS_DIR = EVALUATION_DIR / "results"
