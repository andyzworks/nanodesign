# NanoDesign Progress

Current Stage: STAGE_3
Status: RUNNING

Completed:
- Stage 0: PASS
- Stage 1: PASS
- Stage 2: PASS

Current Goal:
Validate the retained Stage-2 reference recipe independently on 128 fixed
samples for Binder, H3, and RNA without changing any other training variable.

Latest Key Finding:
Binder passes its 128-example task gate at 24K (28.31% validation recovery), and H3
passes at 6K (43.89%). RNA improves continuously from 5.90% untrained recovery to
37.02% at 12K while validation loss falls from 1.8746 to 0.5259. RNA generation
collapse is resolved at 12K: all 8 sequences are distinct and finite, all use at
least three base types, and mean/maximum dominant-token fractions are 67.24%/87.50%.
At median training noise, RNA coordinate RMSD is 1.6435 Angstrom with correct context,
1.6464 shuffled, and 1.7020 detached. Spatial detachment is clearly adverse, but the
target-sequence-shuffle margin remains too weak for the stated Stage-3 gate. The
unchanged RNA run therefore continues to 24K; Stage 3 remains RUNNING.

Next Action:
Complete RNA 24K and its already-queued frozen context/generation audits, then decide
the Stage-3 gate. Binder and H3 require no further Stage-3 training. Do not enter
Stage 4 before `docs/STAGE_3_128_SAMPLE.md` is written with a PASS decision.

Last Updated:
2026-09-05 16:51 CDT
