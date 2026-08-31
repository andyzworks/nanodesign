# NanoDesign v0 frozen training workload sizes

Generated at `2026-08-31T16:33:32.036791+00:00` from the frozen train catalogs.
The audit opens feature-cache databases in SQLite read-only/query-only mode; it does not build cache entries or alter data, filters, splits, model, loss, or sampling. A database is opened only after its finalized SHA-256 sidecar exists and its recorded size matches; partial databases are never opened.

Routing uses the frozen rule: **standard <= 8008; chunked > 8008 model atoms**. Atom statistics cover only valid entries already present in a finalized cache. Uncached rows and rows in a non-finalized database remain `unknown`; lower/upper bounds therefore do not pretend that partial cache coverage is a full-catalog measurement.

Quantiles use linear interpolation (the NumPy `method=linear` convention).

| Task | Train samples | Resolved residues p50 / p95 / max | Cache coverage | Cached atoms p50 / p95 / max | Standard / chunked / unknown | >8008 among cached |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| protein_binder | 40,883 | 49 / 118 / 2023 | 0/40,883 (0.00%) | n/a | 0 / 0 / 40,883 | n/a |
| antibody_h3 | 3,878 | 450 / 803 / 2594 | 0/3,878 (0.00%) | n/a | 0 / 0 / 3,878 | n/a |
| rna | 2,117 | 661 / 1501.8 / 2820 | 2,117/2,117 (100.00%) | 6043 / 8044 / 10620 | 1,997 / 120 / 0 | 5.67% |

## Largest samples by catalog resolved residues

### protein_binder

| Sample | Source | Resolved residues | Cached model atoms | Route |
| --- | --- | ---: | ---: | --- |
| `ppiref50k:6kwy_B_c` | ppiref50k | 2,023 | unknown | unknown |
| `ppiref50k:7nky_B_D` | ppiref50k | 1,296 | unknown | unknown |
| `ppiref50k:6f1t_e_g` | ppiref50k | 1,274 | unknown | unknown |
| `ppiref50k:8imi_0_w` | ppiref50k | 1,207 | unknown | unknown |
| `ppiref50k:8imj_0_J` | ppiref50k | 1,207 | unknown | unknown |
| `ppiref50k:8imj_0_L` | ppiref50k | 1,207 | unknown | unknown |
| `ppiref50k:8imj_0_W` | ppiref50k | 1,207 | unknown | unknown |
| `ppiref50k:8imi_0_E` | ppiref50k | 1,205 | unknown | unknown |
| `ppiref50k:8imi_0_s` | ppiref50k | 1,205 | unknown | unknown |
| `ppiref50k:6sh8_K_Y` | ppiref50k | 1,197 | unknown | unknown |

### antibody_h3

| Sample | Source | Resolved residues | Cached model atoms | Route |
| --- | --- | ---: | ---: | --- |
| `sabdab2:pdb_00009mx1_B_C` | sabdab2 | 2,594 | unknown | unknown |
| `sabdab2:pdb_00007tpr_F_+` | sabdab2 | 2,412 | unknown | unknown |
| `sabdab2:pdb_00007tpr_H_+` | sabdab2 | 2,412 | unknown | unknown |
| `sabdab2:pdb_00007m7e_C_D` | sabdab2 | 2,048 | unknown | unknown |
| `sabdab2:pdb_000011sz_C_D` | sabdab2 | 1,954 | unknown | unknown |
| `sabdab2:pdb_00009e7x_H_L` | sabdab2 | 1,879 | unknown | unknown |
| `sabdab2:pdb_00007v3g_G_H` | sabdab2 | 1,721 | unknown | unknown |
| `sabdab2:pdb_00008y3i_K_L` | sabdab2 | 1,720 | unknown | unknown |
| `sabdab2:pdb_00008y3l_K_L` | sabdab2 | 1,720 | unknown | unknown |
| `sabdab2:pdb_00008y3h_K_L` | sabdab2 | 1,714 | unknown | unknown |

### rna

| Sample | Source | Resolved residues | Cached model atoms | Route |
| --- | --- | ---: | ---: | --- |
| `pdb_rna_target:7wm4:0` | pdb_rna_target_complex | 2,820 | 9,102 | chunked |
| `pdb_rna_target:7k98:0` | pdb_rna_target_complex | 2,508 | 8,872 | chunked |
| `pdb_rna_target:5xuz:0` | pdb_rna_target_complex | 2,504 | 7,216 | standard |
| `pdb_rna_target:8y09:0` | pdb_rna_target_complex | 2,499 | 6,940 | standard |
| `pdb_rna_target:8y07:0` | pdb_rna_target_complex | 2,494 | 6,894 | standard |
| `pdb_rna_target:9drv:0` | pdb_rna_target_complex | 2,477 | 8,642 | chunked |
| `pdb_rna_target:9drt:0` | pdb_rna_target_complex | 2,476 | 8,780 | chunked |
| `pdb_rna_target:9dtf:0` | pdb_rna_target_complex | 2,459 | 8,734 | chunked |
| `pdb_rna_target:9drs:0` | pdb_rna_target_complex | 2,451 | 8,642 | chunked |
| `pdb_rna_target:9dsx:0` | pdb_rna_target_complex | 2,451 | 8,734 | chunked |

## Largest cached samples by model atom count

### protein_binder

| Sample | Source | Resolved residues | Model atoms | Route |
| --- | --- | ---: | ---: | --- |

### antibody_h3

| Sample | Source | Resolved residues | Model atoms | Route |
| --- | --- | ---: | ---: | --- |

### rna

| Sample | Source | Resolved residues | Model atoms | Route |
| --- | --- | ---: | ---: | --- |
| `pdb_rna_target:9nln:0` | pdb_rna_target_complex | 684 | 10,620 | chunked |
| `pdb_rna_target:2xxa:0` | pdb_rna_target_complex | 1,592 | 10,068 | chunked |
| `pdb_rna_target:8iby:0` | pdb_rna_target_complex | 991 | 9,930 | chunked |
| `pdb_rna_target:11hx:0` | pdb_rna_target_complex | 629 | 9,884 | chunked |
| `pdb_rna_target:9lir:0` | pdb_rna_target_complex | 688 | 9,838 | chunked |
| `pdb_rna_target:2v3c:0` | pdb_rna_target_complex | 1,170 | 9,792 | chunked |
| `pdb_rna_target:4xco:0` | pdb_rna_target_complex | 611 | 9,792 | chunked |
| `pdb_rna_target:9liq:0` | pdb_rna_target_complex | 685 | 9,746 | chunked |
| `pdb_rna_target:9lis:0` | pdb_rna_target_complex | 685 | 9,746 | chunked |
| `pdb_rna_target:9lj4:0` | pdb_rna_target_complex | 685 | 9,746 | chunked |

## RNA samples routed to chunked mode in the cache snapshot

All cached RNA train samples above the frozen threshold are listed here.

| Sample | Source | Resolved residues | Model atoms |
| --- | --- | ---: | ---: |
| `pdb_rna_target:9nln:0` | pdb_rna_target_complex | 684 | 10,620 |
| `pdb_rna_target:2xxa:0` | pdb_rna_target_complex | 1,592 | 10,068 |
| `pdb_rna_target:8iby:0` | pdb_rna_target_complex | 991 | 9,930 |
| `pdb_rna_target:11hx:0` | pdb_rna_target_complex | 629 | 9,884 |
| `pdb_rna_target:9lir:0` | pdb_rna_target_complex | 688 | 9,838 |
| `pdb_rna_target:2v3c:0` | pdb_rna_target_complex | 1,170 | 9,792 |
| `pdb_rna_target:4xco:0` | pdb_rna_target_complex | 611 | 9,792 |
| `pdb_rna_target:9liq:0` | pdb_rna_target_complex | 685 | 9,746 |
| `pdb_rna_target:9lis:0` | pdb_rna_target_complex | 685 | 9,746 |
| `pdb_rna_target:9lj4:0` | pdb_rna_target_complex | 685 | 9,746 |
| `pdb_rna_target:11hr:0` | pdb_rna_target_complex | 661 | 9,631 |
| `pdb_rna_target:7utn:0` | pdb_rna_target_complex | 581 | 9,608 |
| `pdb_rna_target:8csz:0` | pdb_rna_target_complex | 678 | 9,608 |
| `pdb_rna_target:8ctl:0` | pdb_rna_target_complex | 585 | 9,608 |
| `pdb_rna_target:11hu:0` | pdb_rna_target_complex | 678 | 9,585 |
| `pdb_rna_target:6mwn:0` | pdb_rna_target_complex | 1,058 | 9,562 |
| `pdb_rna_target:9enf:0` | pdb_rna_target_complex | 712 | 9,562 |
| `pdb_rna_target:8ibz:0` | pdb_rna_target_complex | 997 | 9,516 |
| `pdb_rna_target:7xht:0` | pdb_rna_target_complex | 574 | 9,493 |
| `pdb_rna_target:8vma:0` | pdb_rna_target_complex | 1,056 | 9,424 |
| `pdb_rna_target:7yoj:0` | pdb_rna_target_complex | 1,041 | 9,378 |
| `pdb_rna_target:11ic:0` | pdb_rna_target_complex | 537 | 9,345 |
| `pdb_rna_target:8vm9:0` | pdb_rna_target_complex | 1,050 | 9,286 |
| `pdb_rna_target:4uyj:0` | pdb_rna_target_complex | 521 | 9,274 |
| `pdb_rna_target:8zmi:0` | pdb_rna_target_complex | 867 | 9,263 |
| `pdb_rna_target:8zmj:0` | pdb_rna_target_complex | 865 | 9,263 |
| `pdb_rna_target:8zmk:0` | pdb_rna_target_complex | 867 | 9,263 |
| `pdb_rna_target:10fc:0` | pdb_rna_target_complex | 918 | 9,240 |
| `pdb_rna_target:5on2:0` | pdb_rna_target_complex | 1,887 | 9,171 |
| `pdb_rna_target:8iaz:0` | pdb_rna_target_complex | 547 | 9,125 |
| `pdb_rna_target:7wm4:0` | pdb_rna_target_complex | 2,820 | 9,102 |
| `pdb_rna_target:9nvu:0` | pdb_rna_target_complex | 765 | 9,102 |
| `pdb_rna_target:5ah5:0` | pdb_rna_target_complex | 1,721 | 9,079 |
| `pdb_rna_target:2r8s:0` | pdb_rna_target_complex | 591 | 9,033 |
| `pdb_rna_target:9nzp:0` | pdb_rna_target_complex | 548 | 9,033 |
| `pdb_rna_target:8p7c:0` | pdb_rna_target_complex | 1,539 | 9,010 |
| `pdb_rna_target:2bte:0` | pdb_rna_target_complex | 1,908 | 8,964 |
| `pdb_rna_target:8j12:0` | pdb_rna_target_complex | 964 | 8,941 |
| `pdb_rna_target:8p7b:0` | pdb_rna_target_complex | 1,297 | 8,941 |
| `pdb_rna_target:2fmt:0` | pdb_rna_target_complex | 782 | 8,918 |
| `pdb_rna_target:8dzj:0` | pdb_rna_target_complex | 766 | 8,895 |
| `pdb_rna_target:9k32:0` | pdb_rna_target_complex | 886 | 8,895 |
| `pdb_rna_target:7k98:0` | pdb_rna_target_complex | 2,508 | 8,872 |
| `pdb_rna_target:9lgi:0` | pdb_rna_target_complex | 835 | 8,849 |
| `pdb_rna_target:9tq4:0` | pdb_rna_target_complex | 973 | 8,803 |
| `pdb_rna_target:2deu:0` | pdb_rna_target_complex | 876 | 8,780 |
| `pdb_rna_target:3wfs:0` | pdb_rna_target_complex | 963 | 8,780 |
| `pdb_rna_target:9drt:0` | pdb_rna_target_complex | 2,476 | 8,780 |
| `pdb_rna_target:3wc2:0` | pdb_rna_target_complex | 1,099 | 8,757 |
| `pdb_rna_target:5e6m:0` | pdb_rna_target_complex | 1,171 | 8,734 |
| `pdb_rna_target:9auf:0` | pdb_rna_target_complex | 535 | 8,734 |
| `pdb_rna_target:9dsx:0` | pdb_rna_target_complex | 2,451 | 8,734 |
| `pdb_rna_target:9dtf:0` | pdb_rna_target_complex | 2,459 | 8,734 |
| `pdb_rna_target:9ene:0` | pdb_rna_target_complex | 804 | 8,734 |
| `pdb_rna_target:7l48:0` | pdb_rna_target_complex | 1,185 | 8,688 |
| `pdb_rna_target:8zdr:0` | pdb_rna_target_complex | 590 | 8,688 |
| `pdb_rna_target:9c2k:0` | pdb_rna_target_complex | 811 | 8,688 |
| `pdb_rna_target:9e7d:0` | pdb_rna_target_complex | 811 | 8,688 |
| `pdb_rna_target:9enb:0` | pdb_rna_target_complex | 780 | 8,688 |
| `pdb_rna_target:2zm5:0` | pdb_rna_target_complex | 740 | 8,665 |
| `pdb_rna_target:2zxu:0` | pdb_rna_target_complex | 754 | 8,665 |
| `pdb_rna_target:3foz:0` | pdb_rna_target_complex | 751 | 8,665 |
| `pdb_rna_target:6jdv:0` | pdb_rna_target_complex | 1,179 | 8,665 |
| `pdb_rna_target:2zzn:0` | pdb_rna_target_complex | 813 | 8,642 |
| `pdb_rna_target:9drs:0` | pdb_rna_target_complex | 2,451 | 8,642 |
| `pdb_rna_target:9drv:0` | pdb_rna_target_complex | 2,477 | 8,642 |
| `pdb_rna_target:5hr7:0` | pdb_rna_target_complex | 856 | 8,596 |
| `pdb_rna_target:8okd:0` | pdb_rna_target_complex | 722 | 8,596 |
| `pdb_rna_target:6lvr:0` | pdb_rna_target_complex | 525 | 8,550 |
| `pdb_rna_target:8w2z:0` | pdb_rna_target_complex | 662 | 8,527 |
| `pdb_rna_target:3ndb:0` | pdb_rna_target_complex | 643 | 8,504 |
| `pdb_rna_target:8ygj:0` | pdb_rna_target_complex | 1,913 | 8,504 |
| `pdb_rna_target:9lpc:0` | pdb_rna_target_complex | 781 | 8,504 |
| `pdb_rna_target:7l49:0` | pdb_rna_target_complex | 1,176 | 8,481 |
| `pdb_rna_target:5hr6:0` | pdb_rna_target_complex | 840 | 8,458 |
| `pdb_rna_target:9jfp:0` | pdb_rna_target_complex | 565 | 8,458 |
| `pdb_rna_target:7wju:0` | pdb_rna_target_complex | 740 | 8,435 |
| `pdb_rna_target:9nzs:0` | pdb_rna_target_complex | 860 | 8,412 |
| `pdb_rna_target:9jfo:0` | pdb_rna_target_complex | 562 | 8,389 |
| `pdb_rna_target:6kc8:0` | pdb_rna_target_complex | 1,166 | 8,366 |
| `pdb_rna_target:9nzr:0` | pdb_rna_target_complex | 1,018 | 8,343 |
| `pdb_rna_target:3ivk:0` | pdb_rna_target_complex | 565 | 8,320 |
| `pdb_rna_target:3ivk:1` | pdb_rna_target_complex | 565 | 8,320 |
| `pdb_rna_target:9nzq:0` | pdb_rna_target_complex | 878 | 8,320 |
| `pdb_rna_target:8wt6:0` | pdb_rna_target_complex | 1,373 | 8,297 |
| `pdb_rna_target:8wt7:0` | pdb_rna_target_complex | 1,373 | 8,297 |
| `pdb_rna_target:8wt8:0` | pdb_rna_target_complex | 1,373 | 8,297 |
| `pdb_rna_target:8wt9:0` | pdb_rna_target_complex | 1,385 | 8,297 |
| `pdb_rna_target:8wuv:0` | pdb_rna_target_complex | 1,905 | 8,274 |
| `pdb_rna_target:8zq9:0` | pdb_rna_target_complex | 558 | 8,251 |
| `pdb_rna_target:9why:0` | pdb_rna_target_complex | 1,375 | 8,251 |
| `pdb_rna_target:9t56:0` | pdb_rna_target_complex | 1,377 | 8,228 |
| `pdb_rna_target:9ar4:0` | pdb_rna_target_complex | 1,142 | 8,205 |
| `pdb_rna_target:9nzt:0` | pdb_rna_target_complex | 720 | 8,182 |
| `pdb_rna_target:7wb1:0` | pdb_rna_target_complex | 1,081 | 8,159 |
| `pdb_rna_target:8umf:0` | pdb_rna_target_complex | 1,497 | 8,159 |
| `pdb_rna_target:9kor:0` | pdb_rna_target_complex | 1,350 | 8,159 |
| `pdb_rna_target:9ar6:0` | pdb_rna_target_complex | 1,184 | 8,136 |
| `pdb_rna_target:9ar7:0` | pdb_rna_target_complex | 1,139 | 8,136 |
| `pdb_rna_target:9nzo:0` | pdb_rna_target_complex | 813 | 8,113 |
| `pdb_rna_target:8kag:0` | pdb_rna_target_complex | 1,464 | 8,090 |
| `pdb_rna_target:9whx:0` | pdb_rna_target_complex | 1,372 | 8,090 |
| `pdb_rna_target:6jdq:0` | pdb_rna_target_complex | 1,186 | 8,067 |
| `pdb_rna_target:6je3:0` | pdb_rna_target_complex | 975 | 8,067 |
| `pdb_rna_target:9b0l:0` | pdb_rna_target_complex | 583 | 8,067 |
| `pdb_rna_target:6xjq:0` | pdb_rna_target_complex | 999 | 8,044 |
| `pdb_rna_target:6xjw:0` | pdb_rna_target_complex | 999 | 8,044 |
| `pdb_rna_target:7c7l:0` | pdb_rna_target_complex | 991 | 8,044 |
| `pdb_rna_target:9rw6:0` | pdb_rna_target_complex | 720 | 8,044 |
| `pdb_rna_target:5f9r:0` | pdb_rna_target_complex | 1,477 | 8,021 |
| `pdb_rna_target:5vzl:0` | pdb_rna_target_complex | 1,479 | 8,021 |
| `pdb_rna_target:5wti:0` | pdb_rna_target_complex | 1,097 | 8,021 |
| `pdb_rna_target:6je9:0` | pdb_rna_target_complex | 1,171 | 8,021 |
| `pdb_rna_target:6mcb:0` | pdb_rna_target_complex | 1,602 | 8,021 |
| `pdb_rna_target:6mcc:0` | pdb_rna_target_complex | 1,609 | 8,021 |
| `pdb_rna_target:7way:0` | pdb_rna_target_complex | 1,075 | 8,021 |
| `pdb_rna_target:7waz:0` | pdb_rna_target_complex | 1,067 | 8,021 |
| `pdb_rna_target:7wb0:0` | pdb_rna_target_complex | 912 | 8,021 |
| `pdb_rna_target:8ja0:0` | pdb_rna_target_complex | 1,229 | 8,021 |
| `pdb_rna_target:8wus:0` | pdb_rna_target_complex | 1,881 | 8,021 |

## Reproduce

```bash
PYTHONPATH=src data/envs/rfd3na312/bin/python scripts/audit_training_workload_sizes.py
```

The JSON file is the machine-readable source of truth and includes catalog SHA-256 digests, cache snapshot counters, bounds for the full-catalog chunked fraction, and the ten largest samples by resolved residues and cached model atoms.
