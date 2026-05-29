"""
Train a Physics-Informed Neural Network (PINN) for viscous Burgers equation.
"""

import argparse
import time

import jax
import matplotlib.pyplot as plt
import numpy as np
from exact import compute_viscous_burgers_with_ic_and_periodic_bc_spectral
from helpers import finish_experiment, start_experiment
from loss import (
    InitialConditionLoss,
    PeriodicBoundaryConditionLoss,
    ViscousBurgersEquationLoss,
)
from mlp import MLP
from openinterfaces.interfaces.optim import Optim

jax.config.update("jax_enable_x64", True)


DEFAULT_GRADIENT_TOL = 1e-3


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
        "--gtol",
        default=DEFAULT_GRADIENT_TOL,
        type=float,
        help="Gradient tolerance in inf-norm for the stopping criterion",
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


def get_wrapper_loss_fn(loss_fn_list):
    def wrapper_loss_fn(x, __):
        result = 0.0
        for loss_fn in loss_fn_list:
            result += float(loss_fn(x))
        return result

    return wrapper_loss_fn


def get_wrapper_grad_fn(grad_fn_list):
    def wrapper_grad_fn(x, grad_values, __):
        grad_values[:] = 0.0
        for grad_fn in grad_fn_list:
            grad_values[:] += np.array(grad_fn(x))

    return wrapper_grad_fn


def main():
    args = _parse_args()
    method_name = args.opt

    OUTDIR = start_experiment(args)

    # Mathematical problem parameters
    xleft, xright = 0.0, 2.0
    t0, tfinal = 0.0, 2.0
    x, dx = np.linspace(xleft, xright, num=256, retstep=True)
    t = np.linspace(t0, tfinal, num=21)

    # Viscosity coefficient for the viscous Burgers' equation.
    nu = 0.01 / np.pi

    architecture = [2, 50, 1]
    architecture = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    architecture = [2, 21, 13, 8, 5, 3, 2, 1]
    mlp = MLP(architecture, seed=42, split_key=args.split_key)
    # misfit_loss = MisfitLoss(mlp, x_2d, y)
    # loss_fn = misfit_loss.loss
    eqloss = ViscousBurgersEquationLoss(mlp, given_t=t, given_x=x, nu=nu)
    eqloss_fn = eqloss.loss
    grad_eqloss_fn = jax.grad(eqloss_fn)
    icloss = InitialConditionLoss(mlp, t0, x)
    icloss_fn = icloss.loss
    grad_icloss_fn = jax.grad(icloss_fn)
    bcloss = PeriodicBoundaryConditionLoss(mlp, t, xleft, xright)
    bcloss_fn = bcloss.loss
    grad_bcloss_fn = jax.grad(bcloss_fn)
    wrapper_loss_fn = get_wrapper_loss_fn([eqloss_fn, icloss_fn, bcloss_fn])
    wrapper_grad_fn = get_wrapper_grad_fn(
        [grad_eqloss_fn, grad_icloss_fn, grad_bcloss_fn]
    )

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
    method_name = args.opt
    options = {}
    if args.impl == "scipy_optimize":
        if args.opt == "L-BFGS":
            method_name = "L-BFGS-B"  # Suffix "-B" means "box constraints"
            options = {"ftol": 2.2204460492503131e-09}
        elif args.opt == "BFGS":
            options = {"gtol": args.gtol}
        else:
            raise ValueError()
    elif args.impl == "optim_jl":
        if args.opt == "L-BFGS":
            method_name = "LBFGS"
            options = {
                "f_abstol": 2.2204460492503131e-09,
                "iterations": 30_000,
            }
            if hasattr(args, "linesearch"):
                options["linesearch"] = args.linesearch
        elif args.opt == "BFGS":
            options = {
                "g_abstol": args.gtol,
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
    # pred = mlp.predict(vbeloss.X)
    x_at_t_2 = np.column_stack((tfinal * np.ones(len(x)), x))
    pred = mlp.predict(x_at_t_2)
    np.save(OUTDIR / "predictions.npy", pred)
    print("Prediction shape: ", pred.shape)
    print("Final training loss: ", wrapper_loss_fn(s.x, None))
    print(f"Elapsed time: {toc - tic:.3f} seconds")

    u0 = icloss.ic_map(x)
    exact = compute_viscous_burgers_with_ic_and_periodic_bc_spectral(
        u0, x, dx, tfinal, nu
    )

    plt.plot(x, pred, "-", label=r"$u_{\mathrm{PINN}}$")
    plt.plot(x, exact, "--", label=r"$u_{\mathrm{exact}}$")
    plt.legend(loc="best")
    plt.tight_layout(pad=0.1)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u$")
    plt.savefig(OUTDIR / "predictions.pdf")

    # if args.outdir == "_output":
    #     plt.show()

    finish_experiment(OUTDIR)


if __name__ == "__main__":
    main()
