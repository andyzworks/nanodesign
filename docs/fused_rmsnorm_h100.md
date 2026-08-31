# Fused RMSNorm compatibility check (H100)

Date: 2026-08-31

This check is intentionally isolated from the NanoDesign runtime environment and
does not change the RFD3NA architecture or its configuration.

## Environment and upstream behavior

- NanoDesign environment: Python 3.12, PyTorch 2.7.1+cu128, CUDA runtime 12.8.
- GPU: NVIDIA H100 80GB HBM3.
- Foundry commit: `aad357b776e3c0d6b973080f8f8c4bcf3ed21e40`.
- Apex commit tested: `77a4b7a824c988c62fae477b05bf87727512c309`.
- Foundry tries `apex.normalization.fused_layer_norm.FusedRMSNorm` and falls
  back to `torch.nn.RMSNorm` when Apex cannot be imported.
- The Foundry Python dependencies do not pin Apex. The public README points to
  the `rosettacommons/foundry` Docker image, but does not provide a pinned Apex
  source revision for a native installation.

The cluster's existing Apex Apptainer image is not compatible with the current
NanoDesign environment:

| Component | Cluster Apex image | NanoDesign |
| --- | --- | --- |
| Python | 3.10.12 | 3.12 |
| PyTorch | 2.0.1+cu117 | 2.7.1+cu128 |
| CUDA | 11.7 | 12.8 |

## Isolated native build

The system compiler is GCC 8.5 and the host does not expose a CUDA 12.8
`nvcc`, so a direct build fails. A build succeeds without changing the main
environment when all build dependencies are installed in an isolated Conda
toolkit:

- CUDA compiler and development libraries 12.8 (`nvcc` 12.8.93)
- GCC/G++ 12.4
- Python 3.12 headers
- Apex installed into a separate Python overlay with the upstream
  `APEX_CPP_EXT=1 APEX_CUDA_EXT=1` build flags and `TORCH_CUDA_ARCH_LIST=9.0`

The resulting extension imports with PyTorch 2.7.1+cu128 and runs finite H100
bf16 forward and backward passes. A fresh Foundry import selects
`FusedRMSNorm`, confirming that the compiled extension is actually used.

The overlay additionally needs its matching GCC runtime on
`LD_LIBRARY_PATH`; without it, import fails because the host
`/lib64/libstdc++.so.6` lacks `GLIBCXX_3.4.30`.

## Frozen full-step benchmark

Both variants use the same NanoDesign-Tiny configuration, sample, seed 7,
standard execution, diffusion batch size 4, three warmup steps, and ten timed
steps. These values include the full forward, loss, backward, and optimizer
step.

| Sample | RMSNorm implementation | Step (ms) | Forward (ms) | Backward (ms) | Peak allocated / reserved |
| --- | --- | ---: | ---: | ---: | ---: |
| `ppiref50k:117e_A_B` (756 atoms) | `torch.nn.RMSNorm` | 115.75 | 37.82 | 71.75 | 0.714 / 0.793 GB |
| `ppiref50k:117e_A_B` (756 atoms) | Apex `FusedRMSNorm` | 125.03 | 45.80 | 73.01 | 0.709 / 0.824 GB |
| `sabdab2:pdb_00009nk9_A_+` (1,750 atoms) | `torch.nn.RMSNorm` | 142.87 | 58.62 | 77.99 | 3.034 / 3.202 GB |
| `sabdab2:pdb_00009nk9_A_+` (1,750 atoms) | Apex `FusedRMSNorm` | 150.74 | 65.13 | 79.40 | 3.019 / 3.179 GB |

On these fixed samples, Apex increased full-step time by 8.0% (small) and
5.5% (medium). Peak allocated memory changed by less than 0.02 GB. The Apex
path also emits deprecation/future warnings from its autocast utility under
PyTorch 2.7.1.

## Numerical semantics caveat

Foundry does not pass `eps` to RMSNorm. Apex `FusedRMSNorm` defaults to
`eps=1e-5`, while `torch.nn.RMSNorm` defaults to
`torch.finfo(input.dtype).eps`. The two paths therefore do not have guaranteed
identical numerical semantics, especially for bf16 autocast input. Performance
results must not be treated as authorization to switch the frozen baseline
until this difference is explicitly accepted or upstream Foundry pins matching
semantics.

A direct H100 comparison with identical weights and inputs quantified the
forward difference:

| Input dtype | Input scale | Output relative L2 difference | Maximum absolute difference |
| --- | ---: | ---: | ---: |
| fp32 | 1.0 | 4.92e-6 | 2.15e-5 |
| fp32 | 0.1 | 4.94e-4 | 2.09e-3 |
| fp32 | 0.01 | 4.61e-2 | 1.56e-1 |
| bf16 | 1.0 | 3.05e-5 | 1.95e-3 |
| bf16 | 0.1 | 1.68e-3 | 1.56e-2 |
| bf16 | 0.01 | 4.61e-2 | 1.88e-1 |

The mismatch grows for low-amplitude activations, as expected from the
different epsilon values. This is evidence that the paths are not a transparent
kernel-only substitution under the current upstream defaults.

## Current recommendation

Keep `torch.nn.RMSNorm` as the NanoDesign v0 baseline. The isolated Apex overlay
demonstrates technical compatibility, but it is slower on both representative
samples, provides no material memory reduction, emits PyTorch 2.7 autocast
deprecation warnings, and changes RMSNorm numerical semantics through its
different default epsilon. Do not add Apex to the main environment for v0.
