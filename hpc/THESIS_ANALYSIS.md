MASTER SYNTHESIS (.analysis/hpc/MASTER.md):

Produce ONE standalone markdown document I can paste into another 
chat. It must be readable without access to the source files. 
Structure:

1. PROJECT CONTEXT
   - Problem, dataset, why YOLO, evaluation metrics that matter
   - Infrastructure: SLURM cluster, GPUs, relevant constraints

2. PIPELINE
   - Components and data flow
   - Hyperparameter tuning approach (tune_hyperpara.py mechanics)
   - Key scripts and their roles, in plain prose

3. EXPERIMENTAL TIMELINE
   One subsection per run, in chronological order. Each ~1-2 short 
   paragraphs, not exhaustive:
     - run_id, date
     - HYPOTHESIS (from log; quote if explicit)
     - CHANGE vs prior run (config delta in plain language)
     - RESULT (key metrics + verdict: improvement | regression | inconclusive)
     - LEARNING (what we concluded, what it implied for next run)

4. CROSS-CUTTING FINDINGS
   - What consistently helped
   - What didn't help
   - Surprises / unexpected behavior
   - Failure modes (OOMs, checkpoint issues, naming bugs, etc.)

5. FINAL STATE
   - Best run + its metrics
   - Per-class performance, weakest classes, plausible reasons
   - Known limitations

6. UNRUN IDEAS / FUTURE WORK
   - From the log: things considered but not tried, and why

APPENDIX A: full runs_table as a markdown table.
APPENDIX B: glossary of terms/abbreviations used (so the writing 
            chat doesn't have to guess what e.g. mAP50-95 or 
            tune_hyperpara means in this project).

WRITING RULES FOR MASTER.md:
- Self-contained. No "see file X" references — inline what matters.
- Reasoning over reporting: every config change has a stated WHY.
- Quote the log verbatim where it states intent.
- Mark inferred reasoning explicitly as "(inferred)".
- Target ~8-15k words. If you're going over, compress the timeline 
  entries, not the cross-cutting findings.
- No fabrication. If a run's intent is unknown, say "intent not 
  recorded" rather than inventing one.

use the /grill-me skill to further define the goal of this.

Generate MASTER.md last, after all per-run .md files exist.