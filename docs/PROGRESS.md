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
44.86% in the corrected deterministic centered frame. Under the frozen v2
protocol, Binder 12K reaches 44.81% recovery (40.47% after context-sequence
shuffle; 25.79% after spatially detaching the context) and H3 6K reaches 55.69%
(43.27% shuffled; 36.60% detached). Both generate 8/8 distinct sequences and
therefore pass their Stage-2 gates. RNA 3K reaches 32.77% recovery but ignores
the sequence-shuffle control and its generation is still collapsed (mean
dominant-base fraction 92.30%), so Stage 2 is not yet complete.

Next Action:
Remain in Stage 2. Continue the RNA 32-example reference to 6K while running the
single controlled sequence-only diagnostic, then audit context use and 8-sample
generation at the saved milestones. Do not start Stage 3 before RNA also passes
and docs/STAGE_2_LEARNABILITY.md is complete.

Last Updated:
2026-09-04 01:30 CDT
