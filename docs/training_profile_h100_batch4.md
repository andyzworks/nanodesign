# NanoDesign v0 training profiler

| mode | task | sample_id | bucket | tokens | atoms | I/O ms | parse ms | features ms | H2D ms | forward ms | loss ms | backward ms | optimizer ms | allocated GB | reserved GB | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard | rna | pdb_rna_target:10be:0 | large/large | 403 | 5813 | 111.555 | 9.473 | 55.591 | 6.179 | 941.509 | 37.034 | 636.356 | 34.165 | 31.503 | 32.751 | ok |
| standard | protein_binder | ppiref50k:1z7q_F_g | large/large | 464 | 6496 | 28.003 | 42.049 | 40.884 | 7.523 | 945.212 | 23.501 | 564.815 | 34.285 | 39.327 | 42.000 | ok |
| standard | protein_binder | ppiref50k:8b6j_a_i | large/large | 501 | 7014 | 54.231 | 50.242 | 49.207 | 10.996 | 936.008 | 35.731 | 579.026 | 34.044 | 45.825 | 48.794 | ok |
| standard | protein_binder | ppiref50k:6trc_B_l | large/large | 536 | 7504 | 38.997 | 47.689 | 62.964 | 8.101 | 955.701 | 17.080 | 598.963 | 33.804 | 52.429 | 55.736 | ok |
| standard | protein_binder | ppiref50k:6kac_b_p | large/large | 572 | 8008 | 29.196 | 48.485 | 54.276 | 9.785 | 1264.835 | 55.988 | 567.972 | 35.277 | 59.687 | 63.391 | ok |
| standard | protein_binder | ppiref50k:6kac_C_S | large/large | 636 | 8904 | 29.088 | 48.863 | 55.500 | 10.197 | 1335.096 | 17.060 | 644.117 | 35.513 | 73.755 | 78.660 | ok |
| standard | protein_binder | ppiref50k:8eqm_A_B | large/large | 717 | 10038 | 82.076 | 25.462 | 63.186 | nan | nan | nan | nan | nan | 83.820 | 83.997 | cuda_oom |
| standard | protein_binder | ppiref50k:7ae7_A_B | large/large | 851 | 11914 | 30.237 | 23.098 | 68.333 | nan | nan | nan | nan | nan | 58.773 | 59.184 | cuda_oom |
| chunked | rna | pdb_rna_target:10be:0 | large/large | 403 | 5813 | 182.060 | 9.095 | 53.657 | 6.138 | 12340.021 | 16.210 | 8724.017 | 29.900 | 5.576 | 6.606 | ok |
| chunked | protein_binder | ppiref50k:6kac_b_p | large/large | 572 | 8008 | 71.955 | 48.833 | 52.301 | 9.246 | 12182.446 | 72.517 | 10875.191 | 30.841 | 10.200 | 11.463 | ok |
| chunked | protein_binder | ppiref50k:8eqm_A_B | large/large | 717 | 10038 | 157.891 | 25.954 | 63.008 | 7.819 | 12144.867 | 107.304 | 12638.897 | 30.282 | 15.426 | 17.314 | ok |
| chunked | protein_binder | ppiref50k:7ae7_A_B | large/large | 851 | 11914 | 99.976 | 22.930 | 70.359 | 9.665 | 12152.460 | 102.167 | 14455.983 | 30.770 | 21.250 | 23.878 | ok |

Timings are means over measured repeats. Backward includes the frozen gradient clip; optimizer includes zero-grad and AdamW step. Peak memory includes the model and batch.

## Observed single-H100 boundary

- Largest successful standard sample: 8904 atoms.
- First observed standard OOM: 10038 atoms.
- Largest measured standard sample retaining at least 20% memory headroom: 8008 atoms.
- All measured samples above that conservative boundary completed in chunked mode.
