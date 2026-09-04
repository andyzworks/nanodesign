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
cause is not a label or design-mask misalignment. A one-example Binder control
learns strongly online (recent training recovery about 45% at 900 exposures and
near-zero sequence loss by about 1.5K), while the 32-example 3K runs remain near
10% and collapsed. This identifies inadequate per-example exposure, compounded
by short-horizon EMA lag, as the leading bottleneck.

Next Action:
Remain in Stage 2. Measure 32-example learning curves through 6K/12K/24K/48K
under four controlled Binder recipes while H3/RNA diagnostics finish. Then
apply the minimum successful recipe/exposure to all three tasks and audit
context use and generation collapse. Do not start Stage 3 before the gate.

Last Updated:
2026-09-04 00:38 CDT
