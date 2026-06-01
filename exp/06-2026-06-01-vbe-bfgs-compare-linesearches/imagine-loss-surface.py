import jax
import matplotlib.pyplot as plt
import numpy as np
from exact import compute_viscous_burgers_with_ic_and_periodic_bc_spectral
from helpers import finish_experiment, start_experiment
from loss import (
    InitialConditionLoss,
    PeriodicBoundaryConditionLoss,
    ViscousBurgersEquationLoss,
)
from mlp import MLP
from openinterfaces.interfaces.optim import Optim

jax.config.update("jax_enable_x64", True)


def get_wrapper_loss_fn(loss_fn_list):
    def wrapper_loss_fn(x, __):
        result = 0.0
        for loss_fn in loss_fn_list:
            result += float(loss_fn(x))
        return result

    return wrapper_loss_fn


def main():
    xleft, xright = 0.0, 2.0
    t0, tfinal = 0.0, 2.0
    x, dx = np.linspace(xleft, xright, num=256, retstep=True)
    t = np.linspace(t0, tfinal, num=21)

    # Viscosity coefficient for the viscous Burgers' equation.
    nu = 0.01 / np.pi

    architecture = [2, 50, 1]
    architecture = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    architecture = [2, 21, 13, 8, 5, 3, 2, 1]
    mlp = MLP(architecture, seed=42, split_key=True)
    # misfit_loss = MisfitLoss(mlp, x_2d, y)
    # loss_fn = misfit_loss.loss
    eqloss = ViscousBurgersEquationLoss(mlp, given_t=t, given_x=x, nu=nu)
    eqloss_fn = eqloss.loss
    icloss = InitialConditionLoss(mlp, t0, x)
    icloss_fn = icloss.loss
    bcloss = PeriodicBoundaryConditionLoss(mlp, t, xleft, xright)
    bcloss_fn = bcloss.loss
    wrapper_loss_fn = get_wrapper_loss_fn([eqloss_fn, icloss_fn, bcloss_fn])

    theta = np.asarray(mlp.theta)

    loss_values = []

    theta42_values = np.linspace(-1, 5, num=201)
    for i, th42 in enumerate(theta42_values):
        theta_i = np.copy(theta)
        theta_i[42 * 1] = 1 * th42
        theta_i[42 * 2] = 2 * th42
        theta_i[42 * 3] = 3 * th42
        theta_i[42 * 4] = 4 * th42
        theta_i[42 * 5] = 5 * th42
        theta_i[42 * 6] = 6 * th42
        theta_i[42 * 7] = 7 * th42

        print(f"i = {i}")
        loss_values.append(wrapper_loss_fn(theta_i, "dummy"))

    plt.figure()
    plt.plot(loss_values, "-")
    plt.show()


if __name__ == "__main__":
    main()
