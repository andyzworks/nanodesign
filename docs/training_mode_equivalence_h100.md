# Standard vs chunked numerical comparison

Both modes use identical initialized parameters, the same fixed batch/noise, and H100 bf16.

| bucket | sample | value | max abs | mean abs | relative L2 |
| --- | --- | --- | ---: | ---: | ---: |
| small | ppiref50k:117e_A_B | output.X_L | 0.0078086853 | 0.00051989208 | 2.7608166e-05 |
| small | ppiref50k:117e_A_B | output.sequence_indices_I | 0 | 0 | 0 |
| small | ppiref50k:117e_A_B | output.sequence_logits_I | 0.027832031 | 0.0064749257 | 0.0096679572 |
| small | ppiref50k:117e_A_B | loss.total | 1.7166138e-05 | 1.7166138e-05 | 2.0232771e-06 |
| small | ppiref50k:117e_A_B | loss.coordinate | 1.1444092e-05 | 1.1444092e-05 | 1.3811741e-06 |
| small | ppiref50k:117e_A_B | loss.sequence | 2.8923154e-05 | 2.8923154e-05 | 0.0001456695 |
| medium | sabdab2:pdb_00009nk9_A_+ | output.X_L | 0.0058574677 | 9.3333525e-05 | 1.7433964e-05 |
| medium | sabdab2:pdb_00009nk9_A_+ | output.sequence_indices_I | 0 | 0 | 0 |
| medium | sabdab2:pdb_00009nk9_A_+ | output.sequence_logits_I | 0.0234375 | 0.0041481731 | 0.0066213789 |
| medium | sabdab2:pdb_00009nk9_A_+ | loss.total | 1.4305115e-05 | 1.4305115e-05 | 1.7229129e-06 |
| medium | sabdab2:pdb_00009nk9_A_+ | loss.coordinate | 1.4305115e-05 | 1.4305115e-05 | 1.7289151e-06 |
| medium | sabdab2:pdb_00009nk9_A_+ | loss.sequence | 0 | 0 | 0 |

State-dict key symmetric difference: []. 
Maximum same-seed parameter difference before loading: 0.
