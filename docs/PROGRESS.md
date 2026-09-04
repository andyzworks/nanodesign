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
to target residue relabeling.

Next Action:
Freeze the deterministic 128-sample panels, run Binder/H3/RNA single-task 6K
checkpoints in parallel, then audit loss, recovery, context use, sample-specific
generation, and collapse before deciding whether the Stage-3 gate passes.

Last Updated:
2026-09-04 16:47 CDT
