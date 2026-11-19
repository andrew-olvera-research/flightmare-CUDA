"""Regenerate the README figure with uv run --with matplotlib tools/rl_comparison/plot_release.py."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).with_name("README.md")


def table_rows(heading):
    """Read the published sweep so the figure and numerical tables stay together."""
    lines = EVIDENCE.read_text(encoding="utf-8").splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith(heading))
    rows = []
    for line in lines[first + 2:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def main():
    sweep = table_rows("| Lanes | Broad sweep:")
    grouping = table_rows("| E | A | GPU collection ms")
    fig, (scale, groups) = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    x = np.arange(len(sweep))
    for col, shift, label, color in [(2, -.19, "Collection", "#137168"), (4, .19, "Full PPO", "#334e86")]:
        bars = scale.bar(x + shift, [float(row[col].removesuffix("x")) for row in sweep],
                         width=.36, label=label, color=color)
        scale.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)
    scale.axhline(1, color="#666666", lw=.9, ls="--")
    scale.set(xticks=x, xticklabels=[r[0] for r in sweep], xlabel="Total agents",
              ylabel="CUDA / best tested parallel CPU throughput", ylim=(0, 17),
              title="State-based training speedup")
    scale.legend(frameon=False, loc="upper left")
    cases = sorted((r for r in grouping if int(r[0].replace(",", "")) * int(r[1].replace(",", "")) == 1000),
                   key=lambda r: int(r[0].replace(",", "")))
    bars = groups.barh([f"{r[0]} × {r[1]}" for r in cases], [float(r[2]) for r in cases], color="#137168", height=.52)
    groups.bar_label(bars, fmt="%.3f ms", padding=4)
    groups.invert_yaxis()
    groups.set(xlim=(0, 7), xlabel="GPU time for 32 collection steps (ms)",
               ylabel="Environments × agents", title="Same 1,000 independent agents")
    for axis in (scale, groups):
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(ROOT / "docs/figures/headless-scaling.png", dpi=170)


if __name__ == "__main__":
    main()
