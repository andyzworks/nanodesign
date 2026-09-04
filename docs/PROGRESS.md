# NanoDesign Progress

Current Stage: STAGE_2
Status: PASS

Completed:
- Stage 0: PASS
- Stage 1: PASS
- Stage 2: PASS

Current Goal:
Complete the Stage-2 record, then read it back before entering Stage 3 and
validate the retained recipe independently on 128 fixed samples per task.

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
Read docs/STAGE_2_LEARNABILITY.md and this file, summarize the retained recipe and
rejected experiments, then enter Stage 3. Run Binder/H3/RNA 128-sample validation
in parallel without changing the recipe or evaluator.

Last Updated:
2026-09-04 16:43 CDT
