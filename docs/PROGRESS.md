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
The original learnability.v1 coordinate frame was out of distribution; the
corrected deterministic v2 frame exposes strong Binder and H3 learning. Binder
12K reaches 44.81% recovery (40.47% after context-sequence shuffle; 25.79%
after spatial detachment), while H3 6K reaches 55.69% (43.27% shuffled; 36.60%
detached). Both generate 8/8 distinct sequences and pass their Stage-2 gates.

RNA 6K now reaches 39.12% recovery and its earlier generation collapse is
resolved: all 8 generated sequences are distinct, all coordinates are finite,
and the mean/max dominant-base fractions are 39.70%/47.22%. However, it still
fails the strict context-use gate: shuffled-context recovery is 39.72% and
spatially detached-context recovery is 38.94%. The RNA one-example isolation
control does use context (81.44% correct versus 77.32% shuffled and 60.82%
detached), so the representation and sampler are capable of doing so. RNA-32
sequence-only, increased sequence-loss weight, and online rather than EMA
weights did not solve this gate.

Next Action:
Remain in Stage 2. The RNA-32 reference continuation is at 21.7K/24K samples.
Audit its 12K and final 24K milestones for context dependence, retaining only a
checkpoint that passes both context controls and non-collapse generation. Do
not start Stage 3 before RNA passes and docs/STAGE_2_LEARNABILITY.md is complete.

Last Updated:
2026-09-04 13:12 CDT
