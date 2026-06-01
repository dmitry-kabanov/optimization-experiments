import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTDIR = Path("_output")

dirs = os.listdir(OUTDIR)

for d in dirs:
    dirname = OUTDIR / d

    if os.path.islink(dirname):
        continue

    if dirname.as_posix().endswith(".inprogress"):
        continue

    loss_trace = np.load(dirname / "loss_trace.npy")

    plt.figure()
    plt.semilogy(loss_trace, "-")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.tight_layout(pad=0.1)
    plt.savefig(OUTDIR / d / "loss_trace_semilogy.pdf")
