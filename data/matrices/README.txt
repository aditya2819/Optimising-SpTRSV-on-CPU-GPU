Test Matrices

This folder contains a small real-matrix set downloaded from the SuiteSparse Matrix Collection
in Matrix Market format. These are intended as approximate real-world stand-ins for the
synthetic families used in GPU New, not exact one-to-one matches.

Family mapping used here:

- random_irregular
  - Sandia/ASIC_100k
  - file: random_irregular/ASIC_100k/ASIC_100k.mtx
  - why: irregular sparse circuit matrix, a reasonable stand-in for generic irregular structure

- banded
  - Oberwolfach/LFAT5000
  - file: banded/LFAT5000/LFAT5000.mtx
  - why: narrow local coupling pattern, good proxy for banded behavior

- chain_like
  - Grund/poli
  - file: chain_like/poli/poli.mtx
  - why: very sparse narrow-dependency structure, used here as an approximate chain-like case

- wide_level
  - SNAP/web-Stanford
  - file: wide_level/web-Stanford/web-Stanford.mtx
  - why: web graph structure often exposes broad independent frontiers after ordering/triangular extraction

- block_structured
  - YZhou/circuit204
  - file: block_structured/circuit204/circuit204.mtx
  - why: explicitly blocky circuit-style structure

- power_law
  - SNAP/soc-sign-epinions
  - file: power_law/soc-sign-epinions/soc-sign-epinions.mtx
  - why: social-network style heavy-tail structure, good stand-in for power-law row distributions

Notes:

- The downloaded tarballs are kept alongside the extracted Matrix Market files.
- Some collections contain extra companion files such as rhs or auxiliary matrices.
- These matrices are general sparse matrices, not guaranteed to already be lower triangular.
  For direct SpTRSV testing, we will likely need a preprocessing step:
  - reorder if needed
  - extract/use a triangular part
  - ensure diagonal entries are usable
