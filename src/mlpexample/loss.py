import jax
import jax.numpy as jnp
import numpy as np
from mlp import call


class MisfitLoss:
    def __init__(self, model, given_x, given_y):
        if len(given_y.shape) == 1:
            given_y = jnp.reshape(given_y, (-1, 1))
        assert given_x.shape[0] == given_y.shape[0]

        self.model = model
        self.given_x = given_x
        self.given_y = given_y

    def loss(self, theta):
        self.model.theta = theta

        pred = self.model(self.given_x)
        squares = (pred - self.given_y) ** 2
        assert pred.shape == squares.shape
        misfit = jnp.mean(squares)
        return misfit


class ViscousBurgersEquationLoss:
    r"""Mean squared error of the `model` for the viscous Burgers' equation.

    Parameters
    ----------
    given_t, given_x : np.ndarray of shape (N,)
        Arrays that contain time and spatial points, respectively.
    nu : float
        Viscousity coefficient.
    """

    def __init__(self, model, given_t, given_x, nu):
        self.u = model
        self.nu = nu

        # assert len(given_t.shape) == 1
        # assert len(given_x.shape) == 1
        t_mesh, x_mesh = jnp.meshgrid(given_t, given_x)

        self.X = jnp.column_stack((t_mesh.ravel(), x_mesh.ravel()))

    def loss(self, theta):
        self.u.theta = theta
        """Compute mean squared error of the equation operator"""

        # Define u as a pure function of (t, x) → scalar, with theta fixed
        def u_scalar(tx):
            # tx shape: (2,) — single collocation point [t, x]
            tx_2d = tx[None, :]  # shape (1, 2) — model expects 2-D input
            out = call(tx_2d, theta, self.u.dims)  # shape (1, 1)
            return out[0, 0]  # scalar

        def residual(tx):
            u_val = u_scalar(tx)

            # Gradient w.r.t. the input tx; index 0 = ∂u/∂t, index 1 = ∂u/∂x
            grad_u = jax.grad(u_scalar)(tx)
            u_t = grad_u[0]
            u_x = grad_u[1]

            # Second derivative w.r.t. x: ∂²u/∂x²
            # Differentiate the function tx → ∂u/∂x
            u_x_fn = lambda tx_point: jax.grad(u_scalar)(tx_point)[1]
            u_xx = jax.grad(u_x_fn)(tx)[1]

            return u_t + u_val * u_x - self.nu * u_xx

        residuals = jax.vmap(residual)(self.X)
        eq_reg = jnp.mean(residuals**2)

        return eq_reg

    def u_scalar(self, tx_point):
        tx_2D = tx_point[None, :]
        out = self.u(tx_2D)
        return out[0, 0]
