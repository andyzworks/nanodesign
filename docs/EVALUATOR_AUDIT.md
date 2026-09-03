# NanoDesign Evaluator Audit

Status: **audited and frozen as `nanodesign.learnability.v1`**

Protocol file: [`configs/evaluation/learnability_v1.json`](../configs/evaluation/learnability_v1.json)

Machine-readable audit: `data/runs/nanodesign-v1/evaluator-audit/summary.json`

## Why this audit was needed

The earlier validation path was suitable as a smoke test, but not as a stable
research signal:

- it evaluated only 16 examples per task;
- diffusion timestep, coordinate noise, and sequence corruption were sampled
  during evaluation;
- repeated scores could therefore change even when the checkpoint did not;
- the reported `sequence_loss` is the official training objective after its
  clamp, so it can saturate and conceal large changes in raw sequence CE;
- training-style denoising/recovery metrics and expensive true-generation
  metrics were not clearly separated;
- whole-complex Antibody DockQ could be dominated by the fixed framework and
  antigen rather than by the designed H3 loop;
- RNA self-consistency semantics had not been pinned at the caller boundary.

No task, dataset, model architecture, training loss, or final scientific metric
was added or changed by this audit.

## Frozen evaluation layers

### A. Learnability evaluation

This is the cheap signal used for debugging, controlled comparisons, and future
training-recipe search. It uses:

- 128 fixed validation examples for Protein Binder;
- 128 fixed validation examples for Antibody H3;
- all 83 available RNA validation examples;
- seed 17 and a fixed sample-selection offset;
- fixed diffusion timestep `t = 0.5`;
- fixed coordinate noise and sequence corruption for every sample;
- EMA weights;
- one diffusion sample per validation example;
- no coordinate augmentation.

The protocol pins the dataset catalog hashes and the selected-sample-ID hashes.
Evaluation runs in `model.eval()` and uses deterministic PyTorch algorithms. The
official RFD3 contiguous segment mean is replaced only inside this evaluator by
a mathematically equivalent deterministic implementation because CUDA
`index_reduce(mean)` is nondeterministic. This does not modify training,
generation, parameters, or the RFD3 architecture.

Reported means are:

- total validation loss (lower is better);
- raw sequence cross entropy (lower is better);
- sequence recovery (higher is better);
- coordinate denoising loss and coordinate MSE (lower is better);
- the official smoothed LDDT loss (lower is better).

`sequence_loss` remains in the output for parity with training, but raw sequence
CE is the preferred continuous diagnostic because the official weighted loss is
clamped.

### B. True-generation evaluation

This layer is reserved for milestone and scientific evaluation. It runs the
existing generation and external tools rather than teacher-forced denoising.

| Task | Frozen interpretation |
| --- | --- |
| Protein Binder | Independent structure prediction, confidence/interface metrics, self-consistency RMSD, Rosetta interface analysis, continuous filter inputs, and final pass rate |
| Antibody H3 | H3 AAR and framework-aligned H3 backbone RMSD are primary; whole-complex DockQ is auxiliary |
| RNA | Refold generated sequence independently, compare generated design structure `X` with refold `X'` for scTM/scRMSD, then evaluate the RNA-target complex with DockQ |

Binder evaluation always emits continuous intermediate values even when the
binary success rate is zero. Computational structure/interface metrics are not
equivalent to experimentally measured binding affinity.

## Sanity tests

The audit constructs Perfect, Perturbed, and Broken predictions independently
for all three tasks. The required ordering passed:

| Task / metric | Perfect | Perturbed | Broken | Result |
| --- | ---: | ---: | ---: | --- |
| Binder self-consistency RMSD ↓ | 0.000 | 0.858 | 39.455 | pass |
| H3 AAR ↑ | 1.000 | 0.727 | 0.000 | pass |
| H3 framework-aligned RMSD ↓ | 0.000 | 0.908 | 39.426 | pass |
| H3 whole-complex DockQ ↑ | 1.000 | 0.963 | 0.420 | auxiliary only |
| RNA scTM ↑ | 1.000 | 0.389 | 0.053 | pass |
| RNA scRMSD ↓ | 0.000 | 0.980 | 12.910 | pass |
| RNA-target DockQ ↑ | 1.000 | 0.912 | 0.018 | pass |

The H3 result demonstrates why global DockQ cannot be the main H3 score: even a
broken H3 retains DockQ 0.420 when the much larger fixed complex remains native.

The exact training-loss sanity test also passes `Perfect < Perturbed < Broken`
for total loss, raw sequence CE, coordinate loss, and LDDT loss, and the reverse
ordering for sequence recovery. The clamped training sequence loss saturates for
the latter two cases as expected; this is why the audit exposes raw CE.

## Determinism result

The same 18K checkpoint was evaluated three times on the same 339-example panel
with the same seed. The maximum absolute difference across all reported mean
metrics was exactly **0.0** (required tolerance: `1e-7`). Each run took about
117–119 seconds of GPU evaluation time and peaked at 51.79 GB allocated memory.

## Re-evaluation of existing checkpoints

All values below use the same frozen panel and protocol. They are learnability
diagnostics, not final design-quality claims.

| Task | Budget | Total loss ↓ | Raw seq CE ↓ | Seq recovery ↑ | Coord. loss ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Binder | 0K | 2.9899 | 4.8860 | 3.07% | 2.6031 |
| Binder | 9K | 1.6819 | 4.5621 | 8.67% | 1.3652 |
| Binder | 18K | 1.5654 | 3.4907 | 11.23% | 1.2540 |
| H3 | 0K | 0.7416 | 5.3756 | 2.98% | 0.3548 |
| H3 | 9K | 0.6039 | 3.7273 | 21.84% | 0.2989 |
| H3 | 18K | 0.6046 | 3.8225 | 20.98% | 0.2970 |
| RNA | 0K | 2.0423 | 4.5720 | 1.26% | 1.6726 |
| RNA | 9K | 1.0834 | 3.3751 | 18.89% | 0.8752 |
| RNA | 18K | 0.9687 | 2.1317 | 23.35% | 0.7775 |

The frozen evaluator detects clear 0K→trained learning in every task. Binder
and RNA continue to improve from 9K to 18K. H3 plateaus or slightly regresses on
the frozen panel, so the current 18K checkpoint is not yet sufficient evidence
of a strong unified reference baseline.

## Reproduction

```bash
PYTHONPATH=src:. data/envs/rfd3na312/bin/python scripts/evaluate_learnability.py \
  --protocol configs/evaluation/learnability_v1.json \
  --checkpoint CHECKPOINT.pt \
  --output evaluation.json

PYTHONPATH=src:. data/envs/rfd3na312/bin/python scripts/audit_learnability_sanity.py \
  --output sanity.json

PYTHONPATH=src:. data/envs/rfd3na312/bin/python scripts/audit_true_evaluator_sanity.py \
  --output-dir true-evaluator-sanity
```

The evaluator definition, panel identities, and seeds are frozen from this point
forward. Subsequent experiments may change one training variable at a time, but
must not change this panel or its metric semantics.
