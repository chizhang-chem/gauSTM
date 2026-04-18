# Example Inputs

These `.finp` files are templates, not turnkey jobs.

Each template uses placeholder paths like `/path/to/...`.
Before running a case, replace them with real Gaussian `.fchk` and `.log` files available on your machine.

Recommended entry points:
- `single_state_mo_projection.finp`: one-state STM from a single Gaussian fchk.
- `delta_scf_bardeen_s.finp`: delta-SCF / Dyson STM with an s probe.
- `delta_scf_bardeen_px.finp`: delta-SCF / Dyson STM with a p_x probe.
- `tddft_single_state.finp`: one TDDFT excited state.
- `tddft_state_range_px.finp`: a range of TDDFT states with a p_x probe.
