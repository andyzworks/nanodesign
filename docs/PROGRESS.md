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
The exact 32-example batch audit passed every target, mask, vocabulary,
normalization, coordinate-noise, and motif-conditioning invariant. The root
cause is therefore narrowed to the training recipe/loss coupling rather than a
label or design-mask misalignment. Eight single-variable Stage 2 diagnostics
are currently running on QGPU3021 and QGPU3009.

Next Action:
Remain in Stage 2. Finish the sequence-only, all-valid supervision, Adam, and
AF3-schedule diagnostics; use the prescribed near-clean sequence-only funnel if
needed. Do not start the 128-example stage until all Stage 2 PASS conditions are
met and documented.

Last Updated:
2026-09-04 00:10 CDT
