import functools

import jax
import jax.numpy as jnp
import numpy as np


class MLP:
    """Multilayer perceptron.

    Multi-layer perceptron that accepts d_0-dimensional input
    and returns d_n-dimensional output.

    Parameters
    ----------
    dims : array_like of shape (n+1,).
        List or array of integers that specify the dimensions of layers.
    """

    def __init__(self, dims, seed=42, split_key=True):
        self.dims = tuple(dims)

        # Number of parameters
        n_params = 0

        d_in = dims[0]
        for d in dims[1:]:
            n_params += d_in * d + d
            d_in = d

        assert dims[-1] == 1

        self.random_key = jax.random.key(seed)
        self.split_key = split_key
        self._theta = self._init_params(n_params)
        print("std of parameters vector: ", jnp.std(self._theta))
        print("Dimension of the parameter vector: ", len(self._theta))

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        assert len(self._theta) == len(value)
        self._theta = value

    def _init_params(self, n_params):
        theta = jnp.empty((n_params,))

        if self.split_key:
            key = self.random_key
        else:
            subkey = self.random_key

        # Initialize weights using the Xavier initialization.
        i = 0
        d_in = self.dims[0]
        for d_out in self.dims[1:]:
            if self.split_key:
                key, subkey = jax.random.split(key)
            W_size = d_in * d_out
            theta = theta.at[i : i + d_in * d_out].set(
                jnp.sqrt(1.0 / d_in) * jax.random.normal(subkey, W_size)
            )
            theta = theta.at[i + W_size : i + W_size + d_out].set(0.0)

            i += W_size + d_out
            d_in = d_out

        return theta

    def __call__(self, x):
        # We delegate to the `call` function here,
        # because it is behind JAX's `jit` compiler, hence, must be pure.
        return call(x, self.theta, self.dims)

    def predict(self, x_inp):
        x_2d = np.atleast_2d(x_inp)
        predictions = np.asarray(self(x_2d))

        return predictions


@functools.partial(jax.jit, static_argnames=("dims",))
def call(x, theta, dims):
    z = x
    i = 0
    d_in = dims[0]
    for d_out in dims[1:-1]:
        W_size = d_in * d_out
        b_size = d_out
        W = theta[i : i + W_size].reshape((d_in, d_out))
        b = theta[i + W_size : i + W_size + b_size]
        z = jnp.tanh(z @ W + b)
        i += W_size + b_size
        d_in = d_out

    d_out = dims[-1]
    W_size = d_in * d_out
    b_size = d_out
    W = theta[i : i + W_size].reshape((d_in, d_out))
    b = theta[i + W_size : i + W_size + b_size]
    z = z @ W + b

    return z
