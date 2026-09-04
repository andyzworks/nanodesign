# NanoDesign Progress

Current Stage: STAGE_2
Status: RUNNING

Completed:
- Stage 0: PASS
- Stage 1: PASS

Current Goal:
Find and fix the training-recipe/root-cause problem that prevents stable
memorization and non-collapsed generation on the frozen 32-example Binder, H3,
and RNA panels.

Latest Key Finding:
The original learnability.v1 panel used arbitrary raw PDB coordinate frames,
while official-style RFD3NA training centers/augments coordinates. On the exact
same Binder 12K online checkpoint, recovery is 10.14% in the old frame versus
44.86% in the corrected deterministic centered frame; shuffled context falls
to 38.88%. The model is learning and using context, but the old evaluator hid
the signal. A versioned v2 protocol preserves the panels and targets.

Next Action:
Remain in Stage 2. Re-evaluate learning curves under learnability.v2, finish
H3/RNA one-sample and 32-sample exposure controls, and run context/collapse
audits on the selected checkpoints. Do not start Stage 3 before all three tasks
pass and docs/STAGE_2_LEARNABILITY.md is complete.

Last Updated:
2026-09-04 01:12 CDT
