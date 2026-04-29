import numpy as np
import numpy.testing as npt
from loss import Loss
from mlp import MLP


def test_identity():
    mlp = MLP([1, 1])
    mlp.theta = np.array([1.0, 0.0])

    y = mlp(2.34)
    assert y == 2.34


def test_scale_2x():
    mlp = MLP([1, 1])
    mlp.theta = np.array([2.0, 0.0])

    y = mlp.predict(2.34)
    assert y == 2 * 2.34


def test_linear_function():
    # MLP as kx + b
    k, b = 28.91, 42.18989
    mlp = MLP([1, 1])
    mlp.theta = np.array([k, b])

    x = 2.34
    expected = k * x + b
    y = mlp.predict([x])
    npt.assert_allclose(y, expected, rtol=1e-6, atol=1e-7)


def test_one_hidden_layer():
    w1 = np.array([[1.0, 11.0], [20.0, 32.33], [3.0, 5.0]])
    b1 = np.array([101.0, 22.0, -7])

    w2 = np.array([[2.0, -5.47, 1.2]])
    b2 = np.array([-42.0])

    weights_and_biases = [w1, b1, w2, b2]
    flattened_weights_and_biases = [x.ravel() for x in weights_and_biases]

    x = [2.718, 3.14159]
    h1 = np.tanh(w1 @ x + b1)
    h2 = w2 @ h1 + b2
    desired = np.atleast_2d(h2)

    mlp = MLP([2, 3, 1])
    mlp.theta = np.hstack(flattened_weights_and_biases)

    y = mlp(np.atleast_2d(x))
    np.testing.assert_allclose(y, desired, rtol=1e-7, atol=1e-15)


def test__vectorized_computation_identity():
    mlp = MLP([1, 1])
    mlp.theta = np.array([1.0, 0.0])

    x = np.array([[2.34], [4.89]])
    y = mlp(x)
    np.testing.assert_allclose(y, x, rtol=1e-6, atol=1e-7)
