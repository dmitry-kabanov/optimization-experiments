#!/usr/bin/env python3
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive Agg backend to avoid GUI issues
matplotlib.use("Agg")


def format_gtol(gtol):
    try:
        exponent = int(np.round(np.log10(gtol)))
        if np.isclose(gtol, 10**exponent):
            return rf"$\mathrm{{gtol}} = 10^{{{exponent}}}$"
    except Exception:
        pass
    return f"gtol = {gtol:.0e}"


def main():
    OUTDIR = Path("_output")
    ASSETS_DIR = Path("assets")

    # Ensure assets directory exists
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    if not OUTDIR.exists():
        print(f"Error: Output directory '{OUTDIR}' does not exist.")
        return

    # Dictionary to hold the grouped traces: group_key -> {gtol -> loss_trace}
    groups = {}

    for item in OUTDIR.iterdir():
        # Ignore symlinks
        if item.is_symlink():
            continue
        # Ignore non-directories
        if not item.is_dir():
            continue

        config_path = item / "config.json"
        loss_path = item / "loss_trace.npy"

        if not config_path.exists() or not loss_path.exists():
            continue

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            loss_trace = np.load(loss_path).squeeze()
        except Exception as e:
            print(f"Warning: Failed to load data from {item.name}: {e}")
            continue

        impl = config.get("impl")
        linesearch = config.get("linesearch")
        gtol = config.get("gtol")

        group_key = f"{impl}-{linesearch}" if linesearch is not None else impl

        if group_key not in groups:
            groups[group_key] = {}

        groups[group_key][gtol] = loss_trace

    linestyles_by_gtol = {1e-3: "-", 1e-4: "--", 1e-5: "-."}
    markers = ["o", "s", "v"]

    # Process each group and create a separate figure
    for group_key, gtol_dict in groups.items():
        print(f"Plotting group: {group_key}")

        plt.figure()
        ax = plt.gca()

        # Sort tolerances to ensure consistent line ordering in the plot/legend
        sorted_gtols = sorted(gtol_dict.keys(), reverse=True)  # e.g. 1e-3, 1e-4, 1e-5

        for i, gtol in enumerate(sorted_gtols):
            loss_trace = gtol_dict[gtol]
            label = f"{gtol:.0e}"
            plt.semilogy(loss_trace, label=label, linestyle=linestyles_by_gtol[gtol])
            plt.plot(
                [len(loss_trace) - 1], [loss_trace[-1]], marker=markers[i], color="k"
            )

        plt.xlabel("Iteration")
        plt.ylabel("Loss")

        # Design details: grid, legend, spines
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Legend with clean white background
        plt.legend(loc="upper right")

        plt.tight_layout()

        # Filename template: "impl-linesearch_loss_trace.pdf" in "assets"
        pdf_path = ASSETS_DIR / f"{group_key}_loss_trace.pdf"
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
