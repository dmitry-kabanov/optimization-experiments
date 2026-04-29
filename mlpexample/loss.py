import jax.numpy as jnp


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

        # assert len(given_t.shape) == 1
        # assert len(given_x.shape) == 1
        t_mesh, x_mesh = jnp.meshgrid(given_t, given_x)

        self.X = jnp.column_stack((t_mesh.ravel(), x_mesh.ravel()))

    def loss(self, theta):
        self.u.theta = theta
        """Compute mean squared error of the equation operator"""
        u_t = jax.vmap(jax.grad(self.u)(self.X))
        # u_xx = jax.grad(u_x, 1)(self.X)

        squares = (u_t) ** 2
        # squares = (u_t + self.u * u_x - self.nu * u_xx) ** 2
        eq_reg = jnp.mean(squares)

        return eq_reg
