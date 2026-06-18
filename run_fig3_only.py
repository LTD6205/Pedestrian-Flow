from pathlib import Path

from run_experiments import run_density_wave_plots

run_density_wave_plots(
    outdir=Path("output"),
    quick=False,
    seed_count=20,
)