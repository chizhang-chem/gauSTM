# Validation Summary

This repository does not keep large `.m`, `.png`, or `.log` outputs under version control.
Instead, the table below summarizes representative local validation runs used during development.

| Workflow | Probe | Backend | Grid | Representative output | Local result summary |
| --- | --- | --- | --- | --- | --- |
| CT-TDDFT, state 1 | s | GPU | 201x201 | `Ex001LMN1` | Completed in `25.4 s`, about `1966` points/s |
| CT-TDDFT, states 1-5 | s | GPU | 201x201 | `Ex001..Ex005` | Completed in `26.2 s`, about `1965` points/s |
| CT-TDDFT, states 1-5 | p_x | GPU | 201x201 | `Ex001..Ex005` | Completed in `72.7 s`, about `608` points/s |
| delta-SCF / Dyson, alpha-EA example | p_x | GPU | 201x201 | `Ex000LMN1` | Completed in `66.6 s`, about `647` points/s |
| delta-SCF / Dyson, 10-job anion_3 p_x batch | p_x | GPU | 201x201 each | 10 `.m/.png/.log` outputs | All 10 jobs completed successfully in local validation |

Notes:
- The TDDFT validation runs used a neutral initial state and an anion-triplet final reference state with excited-state determinants read from a Gaussian TDDFT log.
- The delta-SCF validation runs used the standard Dyson route with Gaussian `.fchk` pairs.
- If you want to reproduce these timings, use the same Python environment and GPU stack described in the README.
