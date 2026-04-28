import jax.numpy as jnp


class Loss:
    def __init__(self, model, given_x, given_y):
        if len(given_y.shape) == 1:
            given_y = jnp.reshape(given_y, (-1, 1))
        assert given_x.shape[0] == given_y.shape[0]

        self.model = model
        self.given_x = given_x
        self.given_y = given_y

    def loss_misfit(self, theta):
        self.model.theta = theta

        pred = self.model(self.given_x)
        squares = (pred - self.given_y) ** 2
        assert pred.shape == squares.shape
        misfit = jnp.mean(squares)
        return misfit

    def loss(self, theta):
        return self.loss_misfit(theta)
