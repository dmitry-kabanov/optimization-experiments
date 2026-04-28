import jax.numpy as jnp
import numpy as np
from jax import grad


class MLP:
    """Multilayer perceptron.

    Multi-layer perceptron that accepts d_0-dimensional input
    and returns d_n-dimensional output.

    Parameters
    ----------
    dims : array_like of shape (n+1,).
        List or array of integers that specify the dimensions of layers.
    """

    def __init__(self, dims):
        self.dims = dims

        # Number of parameters
        n_params = 0

        d_in = dims[0]
        for d in dims[1:]:
            n_params += d_in * d + d
            d_in = d

        assert dims[-1] == 1

        self._theta = self._init_params(n_params)

    @property
    def theta(self) -> np.ndarray:
        return self._theta

    @theta.setter
    def theta(self, value):
        assert len(self._theta) == len(value)
        self._theta = value

    def _init_params(self, n_params) -> np.ndarray:
        theta = np.empty((n_params,))

        # Initialize weights using the Xavier initialization.
        i = 0
        d_in = self.dims[0]
        for d_out in self.dims[1:]:
            W_size = d_in * d_out
            theta[i : i + d_in * d_out] = np.sqrt(1.0 / d_in) * np.random.randn(W_size)
            theta[i + W_size : i + W_size + d_out] = 0.0

            i += W_size + d_out
            d_in = d_out

        return theta

    def __call__(self, x):
        x = np.atleast_2d(x)
        assert x.shape[-1] == self.dims[0]

        z = x
        i = 0
        d_in = self.dims[0]
        for d_out in self.dims[1:-1]:
            W_size = d_in * d_out
            b_size = d_out
            W = self._theta[i : i + W_size].reshape((d_in, d_out))
            b = self._theta[i + W_size : i + W_size + b_size]
            z = jnp.tanh(z @ W + b)
            i += W_size + b_size
            d_in = d_out

        d_out = self.dims[-1]
        W_size = d_in * d_out
        b_size = d_out
        W = self._theta[i : i + W_size].reshape((d_in, d_out))
        b = self._theta[i + W_size : i + W_size + b_size]
        z = z @ W + b

        return z

    def predict(self, x_2d):
        assert len(x_2d.shape) == 2
        predictions = self(x_2d)

        return predictions
