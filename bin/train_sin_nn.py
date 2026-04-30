import matplotlib.pyplot as plt
import numpy as np
from jax import grad
from loss import MisfitLoss
from mlp import MLP
from scipy import optimize

x = np.linspace(-2 * np.pi, 2 * np.pi, num=101)
y = np.sin(x)
x_2d = np.reshape(x, (len(x), -1))

mlp = MLP([1, 50, 1])
misfit_loss = MisfitLoss(mlp, x_2d, y)
loss_fn = misfit_loss.loss
grad_loss = grad(loss_fn)

# res = optimize.minimize(
#     loss_fn, x0=mlp.theta, jac=grad_loss, method="L-BFGS-B", options={"gtol": 1e-1}
# )
res = optimize.minimize(
    loss_fn, x0=mlp.theta, jac=grad_loss, method="L-BFGS-B", options={"gtol": 1e-1}
)

assert res.status == 0
print("res = ", res)

mlp.theta = res.x
pred = mlp.predict(x_2d)

plt.plot(x, y, "-")
plt.plot(x, pred, "o")
plt.savefig("assets/plot_sin_func.pdf")
