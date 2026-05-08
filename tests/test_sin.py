import jax
import matplotlib.pyplot as plt
import numpy as np
from loss import MisfitLoss
from mlp import MLP
from scipy import optimize

# Enable float64 instead of float32 by default.
# Normally, ML folks work with float32 numbers, and JAX uses them by default.
jax.config.update("jax_enable_x64", True)


def test_1():
    x, dx = np.linspace(-2 * np.pi, 2 * np.pi, num=101, retstep=True)
    y = np.sin(x)
    x_2d = np.reshape(x, (len(x), -1))

    mlp = MLP([1, 50, 1])
    loss = MisfitLoss(mlp, x_2d, y)
    loss_fn = loss.loss
    grad_loss = jax.grad(loss_fn)

    res = optimize.minimize(
        loss_fn, x0=mlp.theta, jac=grad_loss, method="L-BFGS-B", options={"gtol": 1e-7}
    )

    assert res.status == 0
    print("res = ", res)

    mlp.theta = res.x
    pred = mlp.predict(x_2d)
    pred = np.squeeze(pred)

    print("L2-error: ", np.linalg.norm(y - pred, 2) * np.sqrt(dx))

    assert np.linalg.norm(y - pred, 2) / np.linalg.norm(y) < 1e-2

    plt.plot(x, y, "-")
    plt.plot(x, pred, "o")
    plt.savefig("assets/sin_func.pdf")
