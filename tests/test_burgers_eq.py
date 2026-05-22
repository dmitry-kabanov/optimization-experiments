import jax
import numpy as np
from jax import grad
from loss import ViscousBurgersEquationLoss
from mlp import MLP
from scipy import optimize

# Enable x64 precision
jax.config.update("jax_enable_x64", True)


def test_burgers_pinn():
    x = np.linspace(0, 2, num=3)
    t = np.linspace(0, 2, num=2)

    nu = 0.01 / np.pi

    mlp = MLP([2, 50, 1])
    vbeloss = ViscousBurgersEquationLoss(mlp, given_t=t, given_x=x, nu=nu)
    loss_fn = vbeloss.loss
    grad_loss = grad(loss_fn)

    res = optimize.minimize(
        loss_fn, x0=mlp.theta, jac=grad_loss, method="L-BFGS-B", options={"gtol": 1e-7}
    )

    assert res.status == 0
    print("res = ", res)

    mlp.theta = res.x
    pred = mlp.predict(vbeloss.X)
    assert pred.shape == (len(x) * len(t), 1)

