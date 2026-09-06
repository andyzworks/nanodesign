# NanoDesign Progress

Current Stage: STAGE_4
Status: RUNNING

Completed:
- Stage 0: PASS
- Stage 1: PASS
- Stage 2: PASS
- Stage 3: PASS

Current Goal:
Establish formal Binder-only, H3-only, and RNA-only baselines, beginning at 6K
samples seen per task and using the retained Stage-2 recipe.

Latest Key Finding:
Stage 3 passes. Binder reaches 29.38% frozen-panel recovery at 24K and H3 reaches
42.79% at 6K. RNA reaches 41.18% recovery at 24K with validation loss 0.4757;
8/8 generations are distinct and finite, every sequence uses at least three bases,
and mean/maximum dominant fractions improve to 60.90%/78.79%. At median training
noise, spatial detachment worsens RNA coordinate RMSD from 1.4187 to 1.4911 Angstrom
and changes 10.10% of predictions, establishing 3D-context dependence.

Next Action:
Read the completed Stage-3 and Stage-2 Markdown records, inventory reusable
single-task checkpoints/results, and execute only the missing Stage-4 formal
single-task baseline work starting at 6K exposure per task.

Last Updated:
2026-09-05 20:52 CDT
