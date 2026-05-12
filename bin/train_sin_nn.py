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
        "--split-key",
        action="store_true",
        default=False,
        help="Split random seed (key) for each layer. See the JAX docs for details",
    )
    return parser.parse_args()


args = _parse_args()

OUTDIR = start_experiment(args)

x = np.linspace(-2 * np.pi, 2 * np.pi, num=101)
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


s = Optim("scipy_optimize")
s.set_initial_guess(np.asarray(mlp.theta, dtype=np.float64))
s.set_objective_fn(wrapper_loss_fn)
s.set_grad_fn(wrapper_grad_fn)
s.set_method("L-BFGS-B", {"gtol": 1e-7})
status, message = s.minimize()

print("status = ", status)
print("message = ", message)

mlp.theta = s.x
pred = mlp.predict(x_2d)
pred = np.squeeze(pred)

plt.plot(x, y, "-")
plt.plot(x, pred, "o")
plt.xlabel(r"$x$")
plt.ylabel(r"$\sin(x)$")
plt.tight_layout(pad=0.1)
plt.savefig(OUTDIR / "sin_func.pdf")

finish_experiment(OUTDIR)
