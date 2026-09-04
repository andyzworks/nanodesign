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
Stage 2 passes with the unchanged 6.85M reference architecture and joint recipe.
Binder 12K, H3 6K, and RNA 24K reach 44.81%, 55.69%, and 70.13% recovery on the
frozen overfit panels. All produce 8/8 distinct sequences with finite coordinates.
Binder and H3 respond strongly to both context controls. For RNA at the official
training distribution's median noise, target sequence shuffle worsens design RMSD
from 1.168 Å to 1.212 Å and spatial detachment worsens it to 1.278 Å, proving a
3D context signal even though native sequence recovery is comparatively insensitive
to target residue relabeling. The frozen Stage-3 128-example untrained controls are
now complete (Binder 1.54%, H3 1.83%, RNA 5.27% recovery), and all three matched
single-task training runs have been launched on QGPU3006. Durable interim milestones
reach 28.31% Binder recovery at 24K, 43.89% H3 recovery at 6K, and 26.19% RNA recovery
at 900 samples. Binder now passes its task-level gate: correct recovery is 29.38%
versus 28.71% shuffled and 22.24% detached, coordinate RMSD worsens from 0.296
Angstrom to 0.321/0.477 Angstrom, and 8/8 finite distinct generations have a maximum
dominant-token fraction of 47.06%. H3 also passes its task-level 128-example gate at
6K: correct-context recovery is 42.79% versus 35.03% shuffled and 28.99% detached,
with 8/8 distinct finite generations and no collapse. RNA-900 has a
clear loss/recovery signal, but its median-noise audit still lacks a correct-context
advantage (5.1211 Angstrom correct versus 5.1217 shuffled and 5.0733 detached); the
original RNA run therefore continues toward 3K and 6K.

Next Action:
Complete RNA 3K/6K and its frozen context/generation audits, then decide the Stage-3
gate. Binder and H3 require no further Stage-3 training.

Last Updated:
2026-09-04 18:16 CDT
