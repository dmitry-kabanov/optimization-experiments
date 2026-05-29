import argparse
import time

import jax
import matplotlib.pyplot as plt
import numpy as np
from helpers import finish_experiment, start_experiment
from loss import MisfitLoss
from mlp import MLP
from openinterfaces.interfaces.optim import Optim

jax.config.update("jax_enable_x64", True)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--impl",
        choices=["scipy_optimize", "optim_jl"],
        default="scipy_optimize",
        help="Which implementation from Open Interfaces to use",
    )
    parser.add_argument(
        "--opt",
        choices=["L-BFGS", "BFGS"],
        default="L-BFGS",
        help="What optimization algorithm to use",
    )
    parser.add_argument(
        "--split-key",
        action="store_true",
        default=False,
        help="Split random seed (key) for each layer. See the JAX docs for details",
    )
    parser.add_argument(
        "--linesearch", help="Line-search strategy for Optim.jl optimizers"
    )
    parser.add_argument(
        "--outdir",
        default="_output",
        help="Path to the dir, in which the results subdir will be created",
    )
    return parser.parse_args()


def get_wrapper_loss_fn(loss_fn):
    def wrapper_loss_fn(x, __):
        return float(loss_fn(x))

    return wrapper_loss_fn


def get_wrapper_grad_fn(grad_fn):
    def wrapper_grad_fn(x, grad_values, __):
        grad_values[:] = np.array(grad_fn(x))

    return wrapper_grad_fn


def main():
    args = _parse_args()
    method_name = args.opt
    if method_name == "L-BFGS":
        if args.impl == "scipy_optimize":
            method_name = "L-BFGS-B"
        elif args.impl == "optim_jl":
            method_name = "LBFGS"
        else:
            raise ValueError()

    OUTDIR = start_experiment(args)

    x, dx = np.linspace(-2 * np.pi, 2 * np.pi, num=101, retstep=True)
    y = np.sin(x)
    y_test = np.sin(x + 0.5 * dx)
    x_2d = np.reshape(x, (len(x), -1))

    mlp = MLP([1, 50, 1], seed=42, split_key=args.split_key)
    misfit_loss = MisfitLoss(mlp, x_2d, y)
    loss_fn = misfit_loss.loss
    grad_fn = jax.grad(loss_fn)
    wrapper_loss_fn = get_wrapper_loss_fn(loss_fn)
    wrapper_grad_fn = get_wrapper_grad_fn(grad_fn)

    s = Optim(args.impl)
    s.set_initial_guess(np.asarray(mlp.theta, dtype=np.float64))
    s.set_objective_fn(wrapper_loss_fn)
    s.set_grad_fn(wrapper_grad_fn)

    # Implementations of different algorithms in SciPy and Optim.jl
    # differ in terms of their termination criterion:
    # *BFGS*:
    #   - SciPy accepts only reltol for x in Inf-norm
    #     and absolute tolerance for the gradient in a given norm
    #     (np.Inf by default),
    #   - Optim.jl accepts more but only in Inf-norm.
    # *L-BFGS*:
    #  - SciPy accepts ftol
    #  - Optim.jl accepts more
    # so we need to match options, to compare red apples to similarly red apples.
    if args.impl == "scipy_optimize":
        if args.opt == "L-BFGS":
            options = {"ftol": 2.2204460492503131e-09}
        elif args.opt == "BFGS":
            options = {"gtol": 1e-7}
        else:
            raise ValueError()
    elif args.impl == "optim_jl":
        if args.opt == "L-BFGS":
            options = {
                "f_abstol": 2.2204460492503131e-09,
                "iterations": 30_000,
            }
            if hasattr(args, "linesearch"):
                options["linesearch"] = args.linesearch
        elif args.opt == "BFGS":
            options = {
                "g_abstol": 1e-7,
                "iterations": 30_000,
            }
            if hasattr(args, "linesearch"):
                options["linesearch"] = args.linesearch
        else:
            raise ValueError()
    else:
        raise ValueError()
    s.set_method(method_name, options)

    tic = time.perf_counter()
    status, message = s.minimize()
    toc = time.perf_counter()

    print("status = ", status)
    print("message = ", message)

    mlp.theta = s.x
    pred = mlp.predict(x_2d)
    pred = np.squeeze(pred)

    print("Prediction error, 2-norm: ", np.linalg.norm(y_test - pred, 2) * np.sqrt(dx))
    print(f"Elapsed time: {toc - tic:.3f} seconds")

    plt.plot(x, y, "-")
    plt.plot(x, pred, "o")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\sin(x)$")
    plt.tight_layout(pad=0.1)
    plt.savefig(OUTDIR / "sin_func.png")
    plt.show()

    finish_experiment(OUTDIR)


if __name__ == "__main__":
    main()
