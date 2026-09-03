# RFD3NA Training Recipe Audit

Status: **audited against the pinned public implementation; reference baseline remains
frozen while evaluator and overfit tests run.**

The comparison target is RosettaCommons Foundry commit
`aad357b776e3c0d6b973080f8f8c4bcf3ed21e40`, especially
`models/rfd3na/configs/experiment/rfd3na.yaml` and the configs it composes. The
NanoDesign model is the upstream `rfd3na.model.RFD3.RFD3` class. NanoDesign does not
replace it with a Cartesian-coordinate Transformer.

## Recipe comparison

| Component | Public RFD3NA | Current NanoDesign reference | Classification |
| --- | --- | --- | --- |
| Network family | `RFD3` with token initializer, pair representation, atom attention encoder/decoder, diffusion transformer and EDM sampler | The same public classes and forward path | same as official |
| Geometry handling | RFD3 atom/pair geometry path | Same path; standard and low-memory implementations are selected only by atom count | same as official |
| Atom association | `atom23` in the RFD3NA experiment | `atom23_unk_x_sequence_independent` | same family; NanoDesign-specific explicit unknown design slots |
| Model scale | Full published channel/block counts | 6,849,538 parameters; reduced channels, blocks, keys and recycle count | modified intentionally |
| Recycles | 2 in the RFD3NA experiment | 1 | modified intentionally for tiny scale |
| Design sequence input | Condition transform controls which motif/design sequence is visible | All task design tokens are unknown (`UNK` for protein, `X` for RNA); fixed context sequence is visible | NanoDesign-specific task conditioning |
| Sequence corruption | Mixture of public condition transforms; not a single fixed mask recipe | Complete masking of the declared Binder, H3 or RNA design region; no additional random token corruption | NanoDesign-specific; must be tested by overfit |
| Coordinate diffusion mask | Determined by public condition transform | Only declared design atoms receive EDM noise; context coordinates remain fixed | same conditional-design semantics |
| EDM timestep/noise | `sample_t_edm`, `sample_noise_edm`, `sigma_data=16` | Same public functions and `sigma_data=16` | same as official |
| Coordinate augmentation | `center_option=diffuse`, `sigma_perturb=2.0`, `sigma_perturb_com=1.0`, random rigid augmentation | Same settings implemented in cached-sample reconstruction | same as official |
| Diffusion realizations | 32 per complex | 16 in the frozen 9K/18K reference | modified intentionally for cost |
| Coordinate loss | `DiffusionLoss`: weight 4, sigma 16, LDDT weight 0.25, public atom weights | Same values | same as official |
| Sequence loss | `SequenceLoss`: weight 0.1, `max_t=1` | Same module and values | same as official |
| Sequence-mask normalization | Public loss averages its masked tensor across all tokens | Positive design-mask weights are multiplied by `L/L_design`, so supervision is averaged over design tokens | modified intentionally; prevents fixed context from diluting H3/RNA gradients |
| Unindexed-token loss | `unindexed_t_alpha=0.75` | Module default is not overridden | inert for current data because NanoDesign has no original unindexed tokens |
| Optimizer | Adam, betas `(0.9, 0.95)`, epsilon `1e-8`, no configured weight decay | AdamW, betas `(0.9, 0.95)`, weight decay `1e-4` | possibly important mismatch |
| Learning rate | AF3 schedule, base LR `1.8e-3`, 1K warmup, 0.95 decay per 50K steps | Constant `5e-4` in the frozen reference | possibly important mismatch |
| Gradient clipping | global norm 10 | global norm 10 in the frozen reference | same as official |
| EMA | 0.999 | 0.999; validation/generation load EMA | same as official |
| Self-conditioning / recycling | Public diffusion module with `use_self=true` | Same module, reduced to one recycle | same mechanism; intentionally reduced depth |
| EDM solver | AF3 solver, `s_min=4e-4`, `s_max=160`, `p=7`, gamma 0.8, noise 1.003, step scale 1.5 | Same numerical settings | same as official |
| Sampling steps | 100 | 50 | modified intentionally for cost |
| Classifier-free guidance | Public base sampler enables CFG for optional RASA/donor/acceptor features | Disabled; those optional guided features are not supplied by the three frozen tasks | NanoDesign-specific and intentional |
| Inference coordinates | Regular design uses centered fixed motif and uninitialized diffused coordinates | Fixed context is centered at its atom-slot COM and design coordinates are zeroed before EDM initialization | same regular-design semantics |
| Sequence decoding | Public sequence head output | Argmax restricted to the declared protein or RNA alphabet | NanoDesign-specific validity constraint |

## Findings that affect interpretation

1. Using the official architecture is not equivalent to using the complete official
   recipe. The current optimizer and LR schedule are the clearest material deviations.
2. Full design-region sequence masking is appropriate for the three fixed tasks, but it
   is narrower than the stochastic mixture of public RFD3NA condition transforms.
3. The raw public sequence loss is clipped before weighting. It therefore saturates at
   `0.4` and is not by itself a sufficiently continuous evaluator. NanoDesign now also
   reports the unclipped token cross-entropy and recovery; the training loss is unchanged.
4. Foundry's metric key `mean_lddt` contains a smoothed LDDT **loss**, not an LDDT score.
   The frozen learnability report exposes it as `lddt_loss` with lower-is-better semantics.
5. `unindexed_t_alpha` differs syntactically but cannot affect the current indexed
   Protein/RNA atom23 samples.

## Decision rule

The 6.85M architecture and current reference checkpoint are not changed during the
evaluator audit. If 32-example overfit fails, the first controlled recipe experiment
must address the optimizer/LR mismatch or masking/noising correctness one factor at a
time. Capacity scaling and larger budgets are not justified until that test passes.
