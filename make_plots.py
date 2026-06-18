from pathlib import Path

import numpy as np

from pedflow.model import SimConfig
from pedflow.plots import (
    plot_fig1_hard_body,
    plot_fig2_remote_comparison,
    plot_fig3_time_development,

)


outdir = Path("output")

full_csv = outdir / "full_results.csv"
quick_csv = outdir / "quick_results.csv"

if full_csv.exists():
    results_csv = full_csv
elif quick_csv.exists():
    results_csv = quick_csv
else:
    raise FileNotFoundError(
        "No results CSV found. Expected output/full_results.csv or output/quick_results.csv."
    )

print(f"Using results file: {results_csv}")

# Fig. 1
plot_fig1_hard_body(
    csv_path=str(results_csv),
    png_path=str(outdir / "fig1_hard_body.png"),
)

# Fig. 2
plot_fig2_remote_comparison(
    csv_path=str(results_csv),
    png_path=str(outdir / "fig2_remote_comparison.png"),
)

# Fig. 3
positions_116_path = outdir / "positions_rho_1.16.csv"
positions_121_path = outdir / "positions_rho_1.21.csv"

if not positions_116_path.exists() or not positions_121_path.exists():
    raise FileNotFoundError(
        "Fig. 3 position files not found. Expected "
        "output/positions_rho_1.16.csv and output/positions_rho_1.21.csv."
    )

positions_116 = np.loadtxt(positions_116_path, delimiter=",")
positions_121 = np.loadtxt(positions_121_path, delimiter=",")

plot_fig3_time_development(
    positions=positions_116,
    L=SimConfig().L,
    png_path=str(outdir / "fig3_time_development_rho_1.16.png"),
)

plot_fig3_time_development(
    positions=positions_121,
    L=SimConfig().L,
    png_path=str(outdir / "fig3_time_development_rho_1.21.png"),
)



print("Saved:")
print(outdir / "fig1_hard_body.png")
print(outdir / "fig2_remote_comparison.png")
print(outdir / "fig3_time_development_rho_1.16.png")
print(outdir / "fig3_time_development_rho_1.21.png")