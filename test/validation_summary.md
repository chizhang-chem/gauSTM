# Validation Summary

This repository does not keep large `.m`, `.png`, or `.log` outputs under version control.
Instead, the table below summarizes representative local validation runs used during development.

| Workflow | Probe | Backend | Initial / final basis size | AO overlap size | Grid | Determinant / Dyson scale | Representative output | Local result summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TDDFT, state 1 | s | GPU | `1519 -> 1528` | `1519 x 1528` | `201 x 201` | excited-state determinant count `1` | `Ex001LMN1` | Completed in `25.4 s`, about `1966` points/s |
| TDDFT, states 1-5 | s | GPU | `1519 -> 1528` | `1519 x 1528` | `201 x 201` | determinant counts by state: `1, 3, 3, 1, 1` | `Ex001..Ex005` | Completed in `26.2 s`, about `1965` points/s |
| TDDFT, states 1-5 | p_x | GPU | `1519 -> 1528` | `1519 x 1528` | `201 x 201` | determinant counts by state: `1, 3, 3, 1, 1` | `Ex001..Ex005` | Completed in `72.7 s`, about `608` points/s |
| delta-SCF / Dyson, alpha-EA example | p_x | GPU | `1519 -> 1519` | `1519 x 1519` | `201 x 201` | `NDyson = 1373`, `offset = 146` | `Ex000LMN1` | Completed in `66.6 s`, about `647` points/s |
| delta-SCF / Dyson, 10-job anion_3 p_x batch | p_x | GPU | `1519 -> 1519` each | `1519 x 1519` each | `201 x 201` each | `NDyson = 1373` in the validated alpha-EA cases | 10 `.m/.png/.log` outputs | All 10 jobs completed successfully in local validation |

Notes:
- The TDDFT validation runs used a neutral initial state (`Na=146`, `Nb=145`) and an anion-triplet final reference state (`Na=147`, `Nb=145`) with excited-state determinant components read from a Gaussian TDDFT log.
- The delta-SCF validation runs used the standard Dyson route with Gaussian `.fchk` pairs.
- Representative TDDFT state metadata from the validated set:
  - state 1: `E=0.1286 eV`, `ncfg=1`
  - state 2: `E=0.2904 eV`, `ncfg=3`
  - state 3: `E=0.2942 eV`, `ncfg=3`
  - state 4: `E=0.5000 eV`, `ncfg=1`
  - state 5: `E=0.6302 eV`, `ncfg=1`
- If you want to reproduce these timings, use the same Python environment and GPU stack described in the README.
