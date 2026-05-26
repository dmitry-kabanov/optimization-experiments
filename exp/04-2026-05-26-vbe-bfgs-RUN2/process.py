import json
import os
from pathlib import Path
import pandas as pd

OUTDIR = "_output"

results = []
results_directories = os.listdir(OUTDIR)
results_directories.sort()

for d in results_directories:
    dirname = os.path.join(OUTDIR, d)
    if os.path.islink(dirname):
        continue
    if dirname.endswith(".inprogress"):
        continue
    print("d = ", d)

    config_filename = os.path.join(dirname, "config.json")
    log_filename = os.path.join(dirname, "log.log")
    res = {}
    with open(config_filename, "r") as fh:
        config = json.load(fh)
        impl = config["impl"]
        opt = config["opt"]
        gtol = config["gtol"]
        split_key = config["split_key"]
        res["impl"] = f"{impl:16s}"
        res["gtol"] = f"{gtol:.0e}"
        res["split_key"] = "yes" if split_key else " no"

    with open(log_filename, "r") as fh:
        log_lines = fh.readlines()

    for lline_ in log_lines:
        line = lline_.strip()
        if res["impl"].strip() == "scipy_optimize":
            if line.startswith("nit"):
                niterations = int(line.split(":")[1])
                res["niterations"] = f"{niterations:4d}"
            if line.startswith("nfev"):
                nfuncevals = int(line.split(":")[1])
                res["nfuncevals"] = f"{nfuncevals:4d}"
            if line.startswith("njev"):
                ngradevals = int(line.split(":")[1])
                res["ngradevals"] = f"{ngradevals:4d}"
            if line.startswith("|∇f"):
                grad_norm = float(line.split("=")[1])
                res["final_grad_norm"] = f"{grad_norm:.2e}"
        elif res["impl"].strip() == "optim_jl":
            if line.startswith("result.iterations"):
                niterations = int((line.split("=")[1]))
                res["niterations"] = f"{niterations:4d}"
            if line.startswith("result.f_calls"):
                nfuncevals = int(line.split("=")[1])
                res["nfuncevals"] = f"{nfuncevals:4d}"
            if line.startswith("result.g_calls"):
                ngradevals = int(line.split("=")[1])
                res["ngradevals"] = f"{ngradevals:4d}"
            if line.startswith("result.g_residual"):
                grad_norm = float(line.split("=")[1])
                res["final_grad_norm"] = f"{grad_norm:.2e}"
        else:
            raise ValueError()
        if line.startswith("Final training loss"):
            loss_value = float(line.split(":")[1])
            res["final_loss"] = f"{loss_value:.2e}"
        if line.startswith("Elapsed time"):
            elapsed = line.split(":")[1].strip()
            res["elapsed"] = elapsed.split(" ")[0]

    print(res)
    results.append(res)

# Create DataFrame
df = pd.DataFrame(results)

# Strip whitespace from string values
for col in df.columns:
    df[col] = df[col].astype(str).str.strip()

# Split into two data frames based on split_key
df_false = df[df["split_key"] == "no"].drop(columns=["split_key"])
df_true = df[df["split_key"] == "yes"].drop(columns=["split_key"])

# Set index to impl and gtol, then transpose so they become columns and other properties become rows
df_false_pivoted = df_false.set_index(["impl", "gtol"]).T
df_true_pivoted = df_true.set_index(["impl", "gtol"]).T

# Configure pandas options to print the entire width of the DataFrame nicely
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("=== DataFrame (split_key = False) ===")
print(df_false_pivoted)
print()
print("=== DataFrame (split_key = True) ===")
print(df_true_pivoted)
print()
