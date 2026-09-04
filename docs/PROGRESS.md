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
reach 17.66% Binder recovery at 6K and 14.85% H3 recovery at 900; RNA has written its
300-sample checkpoint and is validating it. The Binder 6K audit is context-sensitive
but remains sequence-biased (64.32% mean dominant-token fraction versus 97.40% at
initialization), so the exact run is continuing to 12K before a Stage-3 decision.

Next Action:
Complete Binder 12K and the running H3/RNA checkpoints, then audit loss, recovery,
context use, sample-specific generation, and collapse before deciding whether the
Stage-3 gate passes.

Last Updated:
2026-09-04 17:23 CDT
