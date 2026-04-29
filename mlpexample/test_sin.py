import matplotlib.pyplot as plt
import numpy as np
from jax import grad
from loss import Loss
from mlp import MLP
from scipy import optimize


def test_1():
    x = np.linspace(-2 * np.pi, 2 * np.pi, num=101)
    y = np.sin(x)
    x_2d = np.reshape(x, (len(x), -1))

    mlp = MLP([1, 50, 1])
    loss = Loss(mlp, x_2d, y)
    loss_fn = loss.loss_misfit
    grad_loss = grad(loss_fn)

    res = optimize.minimize(
        loss_fn, x0=mlp.theta, jac=grad_loss, method="L-BFGS-B", options={"gtol": 1e-7}
    )

    assert res.status == 0
    print("res = ", res)

    mlp.theta = res.x
    pred = mlp.predict(x_2d)

    plt.plot(x, y, "-")
    plt.plot(x, pred, "o")
    plt.savefig("sin_func.pdf")
