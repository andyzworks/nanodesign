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
The frozen reference recipe failed the 32-example gate: Binder reached 9.88%,
H3 5.22%, and RNA 24.44% recovery after 3K task samples; Binder and RNA
generation collapsed. Raising only the constant learning rate from 5e-4 to
1.8e-3 did not help Binder (7.58%).

Next Action:
Remain in Stage 2. Audit batch/target/mask invariants, then run controlled
single-variable recipe diagnostics. Do not start the 128-example stage until all
Stage 2 PASS conditions are met and documented.

Last Updated:
2026-09-03 23:40 CDT
