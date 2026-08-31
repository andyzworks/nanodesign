# NanoDesign v0 training profiler

| mode | task | sample_id | bucket | tokens | atoms | I/O ms | parse ms | features ms | H2D ms | forward ms | loss ms | backward ms | optimizer ms | allocated GB | reserved GB | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard | protein_binder | ppiref50k:117e_A_B | small/small | 54 | 756 | 4.420 | 0.135 | 17.573 | 0.846 | 37.151 | 2.598 | 70.376 | 3.419 | 0.709 | 0.770 | ok |
| standard | antibody_h3 | sabdab2:pdb_00009nk9_A_+ | medium/medium | 125 | 1750 | 4.233 | 0.948 | 20.873 | 0.961 | 40.654 | 2.545 | 74.480 | 3.368 | 3.027 | 3.215 | ok |
| standard | rna | pdb_rna_target:10be:0 | large/large | 403 | 5813 | 14.473 | 6.978 | 51.689 | 1.809 | 111.735 | 4.307 | 196.130 | 3.413 | 31.548 | 33.202 | ok |
| chunked | protein_binder | ppiref50k:117e_A_B | small/small | 54 | 756 | 4.420 | 0.135 | 17.573 | 2.261 | 808.443 | 2.673 | 534.924 | 3.910 | 0.258 | 0.294 | ok |
| chunked | antibody_h3 | sabdab2:pdb_00009nk9_A_+ | medium/medium | 125 | 1750 | 4.233 | 0.948 | 20.873 | 4.812 | 3234.429 | 2.780 | 2061.445 | 5.534 | 0.472 | 0.503 | ok |
| chunked | rna | pdb_rna_target:10be:0 | large/large | 403 | 5813 | 14.473 | 6.978 | 51.689 | 2.509 | 10730.850 | 5.702 | 7007.025 | 27.373 | 2.119 | 2.479 | ok |
| standard | protein_binder | ppiref50k:1z7q_F_g | large/large | 464 | 6496 | 18.668 | 40.325 | 95.776 | 3.443 | 133.139 | 4.970 | 226.674 | 3.842 | 39.371 | 41.324 | ok |
| standard | protein_binder | ppiref50k:8b6j_a_i | large/large | 501 | 7014 | 0.779 | 49.185 | 96.879 | 2.606 | 154.679 | 5.492 | 263.669 | 4.150 | 45.868 | 48.144 | ok |
| standard | protein_binder | ppiref50k:6trc_B_l | large/large | 536 | 7504 | 40.165 | 46.669 | 99.609 | 3.960 | 165.876 | 5.720 | 283.843 | 4.044 | 52.472 | 55.052 | ok |
| standard | protein_binder | ppiref50k:6kac_b_p | large/large | 572 | 8008 | 0.610 | 47.976 | 102.325 | 2.845 | 187.326 | 6.119 | 324.226 | 4.205 | 59.729 | 62.671 | ok |
| standard | protein_binder | ppiref50k:6kac_C_S | large/large | 636 | 8904 | 0.659 | 48.006 | 124.098 | 5.323 | 224.263 | 7.092 | 423.777 | 4.242 | 73.795 | 81.281 | ok |
| standard | protein_binder | ppiref50k:8eqm_A_B | large/large | 717 | 10038 | 14.258 | 25.686 | 109.081 | nan | nan | nan | nan | nan | 83.803 | 84.024 | cuda_oom |
| standard | protein_binder | ppiref50k:7ae7_A_B | large/large | 851 | 11914 | 21.540 | 22.998 | 93.062 | nan | nan | nan | nan | nan | 58.772 | 59.182 | cuda_oom |
| chunked | protein_binder | ppiref50k:1z7q_F_g | large/large | 464 | 6496 | 28.994 | 39.647 | 105.780 | 3.029 | 6554.910 | 5.185 | 4038.720 | 5.412 | 2.478 | 2.877 | ok |
| chunked | protein_binder | ppiref50k:8b6j_a_i | large/large | 501 | 7014 | 69.405 | 49.306 | 103.847 | 16.141 | 10784.312 | 8.913 | 6602.991 | 5.076 | 2.861 | 3.454 | ok |
| chunked | protein_binder | ppiref50k:6trc_B_l | large/large | 536 | 7504 | 27.346 | 45.868 | 98.845 | 15.244 | 10852.394 | 9.102 | 6903.457 | 5.020 | 3.250 | 3.840 | ok |
| chunked | protein_binder | ppiref50k:6kac_b_p | large/large | 572 | 8008 | 35.516 | 48.402 | 109.704 | 16.342 | 10853.163 | 9.671 | 6923.741 | 5.008 | 3.678 | 4.349 | ok |
| chunked | protein_binder | ppiref50k:6kac_C_S | large/large | 636 | 8904 | 56.080 | 48.319 | 53.820 | 3.858 | 10616.385 | 10.406 | 6656.926 | 5.501 | 4.506 | 5.134 | ok |
| chunked | protein_binder | ppiref50k:8eqm_A_B | large/large | 717 | 10038 | 37.461 | 25.041 | 62.648 | 15.752 | 10791.665 | 11.516 | 6654.545 | 5.151 | 5.679 | 6.421 | ok |
| chunked | protein_binder | ppiref50k:7ae7_A_B | large/large | 851 | 11914 | 22.858 | 22.842 | 69.731 | 16.957 | 11106.441 | 13.966 | 6877.468 | 5.138 | 7.927 | 8.860 | ok |

Timings are means over measured repeats. Backward includes the frozen gradient clip; optimizer includes zero-grad and AdamW step. Peak memory includes the model and batch.

## Observed single-H100 boundary

- Largest successful standard sample: 8904 atoms.
- First observed standard OOM: 10038 atoms.
- Largest measured standard sample retaining at least 20% memory headroom: 8008 atoms.
- All measured samples above that conservative boundary completed in chunked mode.
