import jax
import matplotlib.pyplot as plt
import numpy as np
from loss import MisfitLoss
from mlp import MLP
from openinterfaces.interfaces.optim import Optim

jax.config.update("jax_enable_x64", True)

x = np.linspace(-2 * np.pi, 2 * np.pi, num=101)
y = np.sin(x)
x_2d = np.reshape(x, (len(x), -1))

mlp = MLP([1, 50, 1])
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
plt.savefig("assets/sin_func.pdf")
# plt.show()
