import argparse

import jax
import matplotlib.pyplot as plt
import numpy as np
from loss import MisfitLoss
from mlp import MLP
from openinterfaces.interfaces.optim import Optim

from helpers import finish_experiment, start_experiment

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
        choices=["L-BFGS"],
        default="L-BFGS",
        help="What optimization algorithm to use",
    )
    parser.add_argument(
        "--split-key",
        action="store_true",
        default=False,
        help="Split random seed (key) for each layer. See the JAX docs for details",
    )
    return parser.parse_args()


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
x_2d = np.reshape(x, (len(x), -1))

mlp = MLP([1, 50, 1], seed=42, split_key=args.split_key)
misfit_loss = MisfitLoss(mlp, x_2d, y)
loss_fn = misfit_loss.loss
grad_fn = jax.grad(loss_fn)


def wrapper_loss_fn(x, __):
    return float(loss_fn(x))


def wrapper_grad_fn(x, grad_values, __):
    grad_values[:] = np.array(grad_fn(x))


s = Optim(args.impl)
s.set_initial_guess(np.asarray(mlp.theta, dtype=np.float64))
s.set_objective_fn(wrapper_loss_fn)
s.set_grad_fn(wrapper_grad_fn)
if args.impl == "scipy_optimize":
    options = {"gtol": 1e-7}
elif args.impl == "optim_jl":
    options = {"g_abstol": 1e-7}
else:
    raise ValueError()
s.set_method(method_name, options)
status, message = s.minimize()

print("status = ", status)
print("message = ", message)

mlp.theta = s.x
pred = mlp.predict(x_2d)
pred = np.squeeze(pred)

print("L2-error: ", np.linalg.norm(y - pred, 2) * np.sqrt(dx))

plt.plot(x, y, "-")
plt.plot(x, pred, "o")
plt.xlabel(r"$x$")
plt.ylabel(r"$\sin(x)$")
plt.tight_layout(pad=0.1)
plt.savefig(OUTDIR / "sin_func.pdf")

finish_experiment(OUTDIR)
