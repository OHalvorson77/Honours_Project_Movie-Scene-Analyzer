"""
Analysis Comparison Tool for Phase 2

Compares GPT-4o and Claude analyses to:
1. Identify areas of agreement (high confidence findings)
2. Highlight differences (areas needing human review)
3. Generate a unified "consensus" analysis

This cross-validation approach increases reliability and catches
model-specific biases or hallucinations.
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path


def load_analysis(path: str) -> Optional[Dict[str, Any]]:
    """Load an analysis JSON file if it exists."""
    if Path(path).exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def compare_strings(s1: str, s2: str) -> Dict[str, Any]:
    """Compare two string values and assess similarity."""
    if not s1 or not s2:
        return {"match": False, "reason": "missing_value"}
    
    # Simple word overlap comparison
    words1 = set(s1.lower().split())
    words2 = set(s2.lower().split())
    
    if not words1 or not words2:
        return {"match": False, "reason": "empty"}
    
    overlap = len(words1 & words2)
    union = len(words1 | words2)
    similarity = overlap / union if union > 0 else 0
    
    return {
        "match": similarity > 0.3,
        "similarity": round(similarity, 2),
        "gpt4o": s1,
        "claude": s2
    }


def compare_lists(l1: List[str], l2: List[str]) -> Dict[str, Any]:
    """Compare two lists of strings."""
    if not l1 or not l2:
        return {"match": False, "reason": "missing_value"}
    
    set1 = set(item.lower() for item in l1)
    set2 = set(item.lower() for item in l2)
    
    common = set1 & set2
    only_gpt4o = set1 - set2
    only_claude = set2 - set1
    
    return {
        "match": len(common) > 0,
        "common": list(common),
        "only_gpt4o": list(only_gpt4o),
        "only_claude": list(only_claude),
        "gpt4o": l1,
        "claude": l2
    }


def compare_analyses(
    gpt4o_path: str = "scene_analysis.json",
    claude_path: str = "claude_analysis.json"
) -> Dict[str, Any]:

    gpt4o = load_analysis(gpt4o_path)
    claude = load_analysis(claude_path)
    
    if not gpt4o:
        return {"error": f"GPT-4o analysis not found at {gpt4o_path}"}
    if not claude:
        return {"error": f"Claude analysis not found at {claude_path}"}
    
    comparison = {
        "models_compared": {
            "gpt4o": gpt4o.get("_metadata", {}).get("model", "gpt-4o"),
            "claude": claude.get("_metadata", {}).get("model", "claude-3.5-sonnet")
        },
        "scene_overview": {},
        "emotional_arc": {},
        "themes": {},
        "conflict_interpretations": [],
        "summaries": {},
        "agreements": [],
        "differences": []
    }
    
    # Compare scene overview
    if gpt4o.get("scene_overview") and claude.get("scene_overview"):
        gpt4o_overview = gpt4o["scene_overview"]
        claude_overview = claude["scene_overview"]
        
        for key in ["setting", "time_of_day", "atmosphere", "mood"]:
            comp = compare_strings(
                gpt4o_overview.get(key, ""),
                claude_overview.get(key, "")
            )
            comparison["scene_overview"][key] = comp
            
            if comp.get("match"):
                comparison["agreements"].append(f"scene_overview.{key}")
            else:
                comparison["differences"].append(f"scene_overview.{key}")
    
    # Compare themes
    if gpt4o.get("themes") and claude.get("themes"):
        gpt4o_themes = gpt4o["themes"].get("central_themes", [])
        claude_themes = claude["themes"].get("central_themes", [])
        
        theme_comp = compare_lists(gpt4o_themes, claude_themes)
        comparison["themes"] = theme_comp
        
        if theme_comp.get("common"):
            comparison["agreements"].append("themes.central_themes")
    
    # Compare summaries
    comparison["summaries"] = {
        "gpt4o": gpt4o.get("summary", ""),
        "claude": claude.get("summary", "")
    }
    
    # Compare conflict interpretations
    gpt4o_conflicts = {c["timestamp"]: c for c in gpt4o.get("conflict_interpretations", [])}
    claude_conflicts = {c["timestamp"]: c for c in claude.get("conflict_interpretations", [])}
    
    all_timestamps = set(gpt4o_conflicts.keys()) | set(claude_conflicts.keys())
    
    for ts in sorted(all_timestamps):
        gpt4o_interp = gpt4o_conflicts.get(ts, {})
        claude_interp = claude_conflicts.get(ts, {})
        
        comparison["conflict_interpretations"].append({
            "timestamp": ts,
            "gpt4o_interpretation": gpt4o_interp.get("interpretation", "N/A"),
            "claude_interpretation": claude_interp.get("interpretation", "N/A"),
            "both_present": ts in gpt4o_conflicts and ts in claude_conflicts
        })
    
    # Generate consensus analysis
    comparison["consensus"] = generate_consensus(gpt4o, claude, comparison)
    
    # Summary statistics
    total_fields = len(comparison["agreements"]) + len(comparison["differences"])
    agreement_rate = len(comparison["agreements"]) / total_fields if total_fields > 0 else 0
    
    comparison["statistics"] = {
        "total_fields_compared": total_fields,
        "agreements": len(comparison["agreements"]),
        "differences": len(comparison["differences"]),
        "agreement_rate": round(agreement_rate, 2)
    }
    
    return comparison


def generate_consensus(
    gpt4o: Dict[str, Any],
    claude: Dict[str, Any],
    comparison: Dict[str, Any]
) -> Dict[str, Any]:

    consensus = {
        "scene_overview": {},
        "themes": [],
        "emotional_arc": {},
        "summary": "",
        "conflict_interpretations": []
    }
    
    # Scene overview - prefer more detailed descriptions
    gpt4o_overview = gpt4o.get("scene_overview", {})
    claude_overview = claude.get("scene_overview", {})
    
    for key in ["setting", "time_of_day", "atmosphere", "mood"]:
        g_val = gpt4o_overview.get(key, "")
        c_val = claude_overview.get(key, "")
        # Prefer longer (more detailed) description
        consensus["scene_overview"][key] = g_val if len(g_val) >= len(c_val) else c_val
    
    # Themes - merge both
    gpt4o_themes = gpt4o.get("themes", {}).get("central_themes", [])
    claude_themes = claude.get("themes", {}).get("central_themes", [])
    all_themes = list(set(gpt4o_themes + claude_themes))
    consensus["themes"] = all_themes
    
    # Emotional arc - prefer GPT-4o (typically more structured)
    consensus["emotional_arc"] = gpt4o.get("emotional_arc", claude.get("emotional_arc", {}))
    
    # Summary - combine both
    g_summary = gpt4o.get("summary", "")
    c_summary = claude.get("summary", "")
    consensus["summary"] = f"GPT-4o: {g_summary}\n\nClaude: {c_summary}"
    
    # Conflict interpretations - include both perspectives
    for interp in comparison.get("conflict_interpretations", []):
        consensus["conflict_interpretations"].append({
            "timestamp": interp["timestamp"],
            "interpretations": {
                "gpt4o": interp["gpt4o_interpretation"],
                "claude": interp["claude_interpretation"]
            }
        })
    
    return consensus


def save_comparison(comparison: Dict[str, Any], output_path: str = "analysis_comparison.json"):
    """Save the comparison results."""
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved comparison to {output_path}")


def print_summary(comparison: Dict[str, Any]):
    """Print a human-readable summary of the comparison."""
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    
    stats = comparison.get("statistics", {})
    print(f"\nAgreement Rate: {stats.get('agreement_rate', 0) * 100:.0f}%")
    print(f"  - Agreements: {stats.get('agreements', 0)}")
    print(f"  - Differences: {stats.get('differences', 0)}")
    
    print("\n✓ AGREEMENTS:")
    for item in comparison.get("agreements", []):
        print(f"  - {item}")
    
    if comparison.get("differences"):
        print("\n⚠ DIFFERENCES (review recommended):")
        for item in comparison.get("differences", []):
            print(f"  - {item}")
    
    # Theme comparison
    themes = comparison.get("themes", {})
    if themes.get("common"):
        print(f"\n🎭 SHARED THEMES: {', '.join(themes['common'])}")
    
    # Summaries
    summaries = comparison.get("summaries", {})
    print("\n📝 SUMMARIES:")
    print(f"\nGPT-4o: {summaries.get('gpt4o', 'N/A')}")
    print(f"\nClaude: {summaries.get('claude', 'N/A')}")


def main():
    """Run the comparison."""
    print("=" * 60)
    print("Analysis Comparison - Phase 2 Cross-Validation")
    print("=" * 60)
    
    # Check for required files
    if not Path("scene_analysis.json").exists():
        print("\n⚠️  GPT-4o analysis not found (scene_analysis.json)")
        print("Run: python scene_analyzer.py")
        return
    
    if not Path("claude_analysis.json").exists():
        print("\n⚠️  Claude analysis not found (claude_analysis.json)")
        print("Run: python claude_analyzer.py")
        print("\nGenerating comparison with GPT-4o analysis only...")
        
        # Create a placeholder comparison
        gpt4o = load_analysis("scene_analysis.json")
        comparison = {
            "status": "partial",
            "note": "Only GPT-4o analysis available. Run claude_analyzer.py for full cross-validation.",
            "gpt4o_analysis": gpt4o,
            "claude_analysis": None
        }
        save_comparison(comparison)
        return
    
    # Run comparison
    print("\nComparing GPT-4o and Claude analyses...")
    comparison = compare_analyses()
    
    # Save and display results
    save_comparison(comparison)
    print_summary(comparison)
    
    print("\n" + "=" * 60)
    print("Full comparison saved to analysis_comparison.json")
    print("=" * 60)


if __name__ == "__main__":
    main()



