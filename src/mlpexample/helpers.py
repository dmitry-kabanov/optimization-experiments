import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def start_experiment(args: argparse.Namespace) -> Path:
    """
    Creates a timestamped output directory based on command-line arguments.
    """
    # 1. Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%dT%H.%M.%S")

    # 2. Serialize and sort arguments alphabetically
    # We convert the Namespace to a dict to sort by key
    arg_dict = vars(args)
    sorted_params = []
    for key_orig in arg_dict.keys():
        if key_orig == "outdir":
            continue
        key = key_orig.replace("_", "-")
        val = arg_dict[key_orig]
        # Format boolean flags or None types for cleaner folder names
        val_str = str(val).lower() if isinstance(val, bool) else str(val)
        sorted_params.append(f"{key}={val_str}")

    # Join params with underscores
    params_str = "_".join(sorted_params)

    # 3. Construct the path
    outdir = f"{timestamp}_{params_str}.inprogress"
    if hasattr(args, "outdir"):
        outdir = Path(args.outdir) / outdir
    else:
        outdir = Path("_output") / outdir

    # 4. Check existence and create
    if outdir.exists():
        raise RuntimeError(f"Directory already exists: {outdir}")

    outdir.mkdir(parents=True, exist_ok=False)

    with open(outdir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    return outdir


def finish_experiment(outdir: Path) -> Path:
    if not outdir.name.endswith(".inprogress"):
        raise RuntimeException(f"Cannot find outdir with name '{outdir}'")

    new_name = outdir.name.replace(".inprogress", "")
    new_path = outdir.with_name(new_name)

    finalized_outdir = outdir.rename(new_path)
    print(f"Experiment directory is {finalized_outdir.name}")

    symlink_path = new_path.parent / "current"

    if symlink_path.exists():
        symlink_path.unlink()

    os.symlink(finalized_outdir.name, symlink_path)
