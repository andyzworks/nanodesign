# NanoDesign

NanoDesign is a multi-task framework for conditional biomolecular design. It uses one task-conditioned SE(3)-equivariant model to jointly generate residue-level structure and sequence for protein binders, antibody CDRs, and RNA.

The current implementation is a research prototype and benchmark scaffold. It provides an end-to-end path from versioned molecular examples to corruption, training, hidden-label evaluation, and conditional sampling. Its present output is a coarse residue-frame representation with three canonical anchor atoms per residue; it is not yet a full-atom molecular design system.

## Supported design tasks

| Task | Fixed context | Generated region | Output |
| --- | --- | --- | --- |
| Peptide/protein binder design | Target protein | Binder residues | Binder tokens, residue translations, rotations, and anchors |
| Antibody design | Antigen and antibody framework | CDR residues | CDR tokens, residue translations, rotations, and anchors |
| RNA design | Optional task metadata; the current RNA task is unconditional | RNA residues | RNA tokens, coarse residue translations, rotations, and anchors |

All tasks share one model and one data contract. `task_id`, polymer type, molecular role, chain, entity, position, corruption time, and design mask tell the shared trunk how each residue should be interpreted.

## Framework architecture

```mermaid
flowchart LR
    A[Raw structures] --> B[Task-specific preprocessing]
    B --> C[Versioned NanoDesign examples]
    C --> D[Joint frame and token corruption]
    D --> E[Task-conditioned node and pair encoders]
    E --> F[Shared IPA / SE(3) trunk]
    F --> G1[Translation and rotation heads]
    F --> G2[Polymer-aware token head]
    G1 --> H[Canonical anchor reconstruction]
    G1 --> I[Joint task-balanced loss]
    G2 --> I
    H --> I
    F --> J[Iterative conditional sampler]
    J --> K[NPZ and anchor-atom PDB outputs]
```

### 1. Versioned data contract

Each example describes a complete design problem rather than only a coordinate tensor. Important fields include:

- task, polymer, role, chain, entity, and source residue identifiers;
- clean residue tokens with disjoint protein and RNA vocabularies;
- residue-frame translations and SO(3) rotations;
- three observed anchor atoms and an anchor-validity mask;
- `design_mask`, which exactly identifies the residues the model may change;
- `res_mask`, which distinguishes real residues from batch padding.

Datasets are stored as immutable NPZ examples accompanied by a JSONL manifest, source lock, per-file SHA256 checksums, and a whole-dataset fingerprint. This makes data changes visible and allows checkpoints to be tied to a specific dataset version.

### 2. Joint corruption process

NanoDesign trains the model to recover clean structure and sequence from a shared noisy state:

- designed translations interpolate between centered Gaussian noise and clean coordinates;
- designed rotations interpolate on SO(3) between random and clean frames;
- designed sequence tokens are replaced with a mask token according to corruption time;
- fixed-context tokens and frames remain unchanged.

The same design region is used for structure and sequence, preventing a task from silently optimizing a different residue subset for each modality.

### 3. Task-conditioned encoders

The node encoder combines embeddings for:

- corrupted residue tokens;
- task identity;
- protein or RNA polymer identity;
- target, binder, antigen, framework, CDR, or RNA role;
- chain and entity identity;
- residue position;
- translation, rotation, and token corruption times;
- the design mask.

The pair encoder adds relative residue positions, current inter-residue distance bins, same-chain/entity/polymer/role indicators, and pairwise design-state indicators.

### 4. Shared SE(3) trunk

All three tasks use one Invariant Point Attention trunk. Each block performs:

1. geometry-aware invariant point attention over the current residue frames;
2. masked sequence-style transformer communication;
3. a structure transition;
4. an SE(3) residue-frame update;
5. an optional pair-feature update for the next block.

Frame updates are multiplied by `design_mask`, so fixed target, antigen, and framework frames are protected inside every model block rather than repaired only after prediction.

### 5. Output heads and objective

The model predicts clean residue translations, rotations, canonical anchor coordinates, and unified token logits. Invalid protein/RNA token classes are removed during generation.

The training objective combines:

- translation loss;
- SO(3) geodesic rotation loss;
- anchor-coordinate loss;
- masked-token cross-entropy.

Losses are averaged per example and then macro-averaged across tasks represented in the batch. This prevents a task with more residues or examples from automatically dominating the objective.

### 6. Conditional generation

At inference time, NanoDesign initializes only the designed region from translation, rotation, and token priors. The sampler repeatedly predicts the clean endpoint and advances the noisy frames toward that prediction. Fixed context is restored after every integration step. The final token distribution is masked to the vocabulary of the residue's polymer type.

Generated results can be written as:

- NPZ files containing tokens, translations, rotations, and anchors;
- PDB files containing canonical anchor atoms for visualization and downstream processing.

## What the framework currently provides

- One shared model for binder, antibody CDR, and RNA design.
- Joint sequence and coarse-structure generation.
- Strict fixed-context semantics during corruption, model updates, and sampling.
- Polymer-safe protein/RNA vocabularies.
- Deterministic preprocessing and tamper-detecting dataset fingerprints.
- Single-GPU and distributed data-parallel training.
- Parameter, optimizer-step, sample, and residue budgets for benchmark tracks.
- Checkpointing, deterministic stochastic-input replay, resume integrity checks, and learning-curve AUC.
- A minimal submission interface with `build_model`, `build_optimizer`, and `run_batch` entry points.
- A hidden-label scoring path that removes clean design labels before submission code receives a batch.
- Metrics for joint loss, translation and anchor error, rotation error, token recovery, interface contacts, and fixed-context drift.
- End-to-end prediction with trajectory-capable iterative sampling.

## Runtime and benchmark layers

NanoDesign separates four concerns:

| Layer | Responsibility |
| --- | --- |
| Dataset layer | Validate examples, preserve provenance, collate padded batches, and detect tampering |
| Model layer | Encode task/sequence/geometry context and predict clean frames and tokens |
| Flow layer | Corrupt training examples and integrate conditional samples |
| Benchmark layer | Enforce track budgets, isolate public inputs, score hidden labels, and record reproducibility metadata |

The public runtime exposes corrupted coordinates and tokens plus task/context metadata. It does not intentionally expose clean designed tokens, translations, rotations, anchor coordinates, or hidden scores to submission code.

## Current validation status

The implementation has been exercised with:

- unit and contract tests for all three task types;
- joint forward/backward training;
- dataset-tampering detection;
- public-batch label filtering;
- fixed-context preservation and polymer-valid sampling;
- multi-GPU H100 training, checkpoint/resume, evaluation, and prediction;
- a three-structure public smoke panel covering binder, antibody, and RNA paths.

The public panel is deliberately an end-to-end smoke/overfit test. Improvements on this panel demonstrate that the software path can learn, score, resume, and sample; they do not demonstrate generalization to unseen molecular systems.

## Present limitations

- No large, leakage-controlled, cluster-disjoint train/validation/test corpus is included yet.
- Outputs are residue frames plus three canonical anchors, not full protein side chains, antibody atoms, RNA atoms, or torsions.
- Free sampling is substantially less accurate than fixed-time denoising in the current short smoke run.
- Rotation learning remains weak in the current baseline.
- Antibody CDR windows in the smoke preset are deterministic approximations rather than production ANARCI/IMGT numbering.
- No external folding, docking, self-consistency, energetic, developability, RNA secondary-structure, or wet-lab validation has been established.
- The in-process submission runtime validates the data interface but is not an adversarial security sandbox. A public competition should use restricted containers and an external hidden-data service.
- Hardware-normalized wall-clock or FLOP accounting is not yet part of track enforcement.

## Recommended next milestones

1. Build auditable, cluster-disjoint datasets for all three tasks.
2. Close the official trainer/submission API loop so arbitrary submissions use the same budgeted training path.
3. Strengthen output-contract validation for vocabulary size, valid SO(3) matrices, and frame/anchor consistency.
4. Add production antibody numbering and task-specific cropping and augmentation.
5. Improve SO(3) flow and require free-sampling metrics to improve across multiple seeds.
6. Add polymer-specific torsion/full-atom decoders and stereochemical losses.
7. Add external structure, docking, self-consistency, and task-specific quality evaluation.

## Project status

NanoDesign should currently be treated as a strong engineering prototype and experimental baseline: the framework is complete enough to run reproducible multi-task design experiments, while scientific claims should wait for leakage-controlled data, stronger free sampling, full-atom outputs, and external validation.
