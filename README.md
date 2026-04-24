# Final Project Repo Layout

This copy is arranged like a normal repository so it can be cloned and run directly.

## Layout

- `src/cpu`: CPU benchmark and kernel sources
- `src/gpu`: GPU benchmark and kernel sources
- `scripts/cpu`: local CPU build and run helpers
- `scripts/gpu`: local GPU build and run helpers
- `scripts/explorer`: Explorer batch scripts kept separate from local scripts
- `data/matrices`: all benchmark matrices
- `results`: collected result tables
- `docs`: progress reports and summary notes

## Local Usage

From the repository root:

```bash
bash scripts/cpu/run_benchmark.sh real
bash scripts/cpu/run_benchmark.sh families

bash scripts/gpu/run_benchmark.sh real
bash scripts/gpu/run_benchmark.sh families
```

CPU helper scripts:

```bash
bash scripts/cpu/run_cpu_solve_only_tables.sh
bash scripts/cpu/run_vtune_per_kernel_profiles.sh web-Stanford
```

GPU helper scripts:

```bash
bash scripts/gpu/run_local_full_profile.sh
```

## GPU Architecture Selection

Local GPU builds use `CUDA_ARCH`.

Examples:

```bash
CUDA_ARCH=60 bash scripts/gpu/run_benchmark.sh real
CUDA_ARCH=70 bash scripts/gpu/run_benchmark.sh real
CUDA_ARCH=90 bash scripts/gpu/run_benchmark.sh real
```

## Explorer Usage

Explorer batch scripts live under `scripts/explorer`.

Examples:

```bash
sbatch scripts/explorer/cpu/CPU_Run_Timings.script
sbatch scripts/explorer/gpu/p100/GPU_Run_Timings.script
sbatch scripts/explorer/gpu/h200/GPU_Run_Timings.script
```

These scripts call the repo-local runners and use the repo `data/matrices` paths, so cloning the repo preserves the benchmark layout.
