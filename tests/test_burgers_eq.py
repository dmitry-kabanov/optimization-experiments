import matplotlib.pyplot as plt
import numpy as np
from jax import grad
from loss import ViscousBurgersEquationLoss
from mlp import MLP
from scipy import optimize

x = np.linspace(0, 2, num=3)
t = np.linspace(0, 2, num=2)

nu = 0.01 / np.pi

mlp = MLP([2, 50, 1])
vbeloss = ViscousBurgersEquationLoss(mlp, x, t, nu)
loss_fn = vbeloss.loss
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
plt.savefig("fig_burgers_eq.pdf")
