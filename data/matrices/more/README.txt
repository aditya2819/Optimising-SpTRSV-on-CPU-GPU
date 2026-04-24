Additional Matrix Market Collection

This folder contains extra real matrices downloaded from SuiteSparse Matrix Collection
for future testing beyond the first family-mapped set in `Test Matrices/`.

Primary matrices

These are the main `.mtx` files that should be treated as actual matrix inputs:

1. `Boeing__msc04515/msc04515/msc04515.mtx`
2. `Boeing__pwtk/pwtk/pwtk.mtx`
3. `GHS_indef__helm2d03/helm2d03/helm2d03.mtx`
4. `GHS_psdef__apache1/apache1/apache1.mtx`
5. `HB__bcsstk13/bcsstk13/bcsstk13.mtx`
6. `HB__bcsstk14/bcsstk14/bcsstk14.mtx`
7. `HB__bcsstk17/bcsstk17/bcsstk17.mtx`
8. `HB__bcsstk18/bcsstk18/bcsstk18.mtx`
9. `HB__bcsstk27/bcsstk27/bcsstk27.mtx`
10. `HB__bcsstm22/bcsstm22/bcsstm22.mtx`
11. `Hamm__scircuit/scircuit/scircuit.mtx`
12. `Norris__fv1/fv1/fv1.mtx`
13. `Norris__fv2/fv2/fv2.mtx`
14. `Norris__fv3/fv3/fv3.mtx`
15. `Pajek__EPA/EPA/EPA.mtx`
16. `SNAP__ca-GrQc/ca-GrQc/ca-GrQc.mtx`
17. `SNAP__email-Enron/email-Enron/email-Enron.mtx`
18. `SNAP__roadNet-CA/roadNet-CA/roadNet-CA.mtx`
19. `SNAP__wiki-Vote/wiki-Vote/wiki-Vote.mtx`
20. `Sandia__ASIC_680k/ASIC_680k/ASIC_680k.mtx`

Companion / non-primary files

These are not separate benchmark matrices for our sweep:

- `Hamm__scircuit/scircuit/scircuit_b.mtx`
- `SNAP__ca-GrQc/ca-GrQc/ca-GrQc_nodename.mtx`
- `Pajek__EPA/EPA/EPA_nodename.txt`

Notes

- Most folders also keep the original `.tar.gz` archive beside the extracted matrix.
- Some earlier attempted downloads failed or were too large; those partial archives can
  remain in the tree but are not part of the curated primary list above.
- For SpTRSV use, these matrices will still need the same preprocessing path we already
  use elsewhere: Matrix Market load, lower-triangular extraction, and diagonal fixing
  if needed.
