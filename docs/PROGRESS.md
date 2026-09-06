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
The existing formal single-task 6K checkpoints match the retained recipe. Frozen
large-panel recovery is 15.63% Binder, 39.21% H3, and 30.28% RNA, all clearly above
initialization. H3 generation is non-collapsed. Binder remains highly concentrated
(91.30% maximum dominant-token fraction), while RNA still collapses (five of eight
generations are homopolymers; 95.25%/100% mean/maximum dominance). Stage 4 therefore
cannot pass at 6K.

Next Action:
Continue only Binder-only and RNA-only from their exact 6K checkpoints to 12K with
the unchanged retained recipe. Re-run frozen learnability, context, generation, and
task-specific evaluation at 12K before deciding whether either task needs 18K. Do
not retrain H3 unless later evidence requires it.

Last Updated:
2026-09-05 21:36 CDT
