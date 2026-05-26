import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid


def compute_viscous_burgers_with_ic_and_periodic_bc_spectral(u0, x, dx, t, nu):
    """Compute the exact solution of viscous Burgers' equation via Cole-Hopf transform.

    Compute the exact solution of viscous Burgers' equation
    via Cole-Hopf transform that transforms the PDE
    to a heat equation,
    that along with the periodic boundary conditions
    amounts to using Fourier expansion to find the solution
    and then converting back to the variable of the Burgers' equation.

    Parameters:
    u0 : ndarray
        Initial condition array evaluated at x.
    x  : ndarray
        Grid points.
    t  : float
        Time at which solution u(t, x) will be evaluated.
    nu : float
        Viscosity coefficient.

    Returns:
    u : ndarray
        The solution u(t, x) at the requested time `t` on the grid `x`.
    """
    N = len(x)
    L = x[-1]  # Domain is assumed to be x \in [0; L].

    # 1. Extract the spatial mean (conserved quantity)
    u_bar = np.mean(u0)
    v0 = u0 - u_bar  # Zero-mean velocity field

    # Compute initial condition, which is an integral with the variable upper bound.
    # initial=0.0 ensures the output size matches x
    V0 = cumulative_trapezoid(v0, x, initial=0.0)

    # Enforce strict periodicity to avoid Gibbs ringing in the FFT.
    # V0[-1] should be exactly 0 analytically, but floating point
    # accumulation errors require a tiny linear drift correction.
    V0 -= x * (V0[-1] / (L - dx))

    # Cole-Hopf transformation
    phi0 = np.exp(-V0 / (2 * nu))

    # We use FFT to solve the resultant heat equation with periodic boundary
    # conditions.
    phi_hat0 = np.fft.fft(phi0)

    # Multiplying by 2pi to convert from linear frequencies to angular frequencies.
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

    # Exact time evolution in Fourier space
    # The real part (-nu * k**2 * t) handles diffusion.
    # The imaginary part (-1j * k * u_bar * t) perfectly translates
    # the field to account for the advective mean u_bar.
    exponent = -nu * (k**2) * t - 1j * k * u_bar * t
    phi_hat_t = phi_hat0 * np.exp(exponent)

    # Compute spatial derivative in Fourier space
    phi_x_hat_t = 1j * k * phi_hat_t

    # Invert Fourier Transform back to physical space
    phi_t = np.fft.ifft(phi_hat_t).real
    phi_x_t = np.fft.ifft(phi_x_hat_t).real

    # Invert Cole-Hopf and re-add the mean
    v_t = -2 * nu * (phi_x_t / phi_t)
    u_t = v_t + u_bar

    return u_t


if __name__ == "__main__":
    x, dx = np.linspace(0, 2, 256, endpoint=False, retstep=True)
    u0 = 0.5 - 0.25 * np.sin(np.pi * x)
    nu = 0.01 / np.pi
    tfinal = 2.0

    # Solve
    u_at_t1 = compute_viscous_burgers_with_ic_and_periodic_bc_spectral(
        u0, x, dx, 0.5 * tfinal, nu
    )
    u_at_t2 = compute_viscous_burgers_with_ic_and_periodic_bc_spectral(
        u0, x, dx, tfinal, nu
    )

    plt.plot(x, u0, "--", label="u0")
    plt.plot(x, u_at_t1, "-.", label="u1")
    plt.plot(x, u_at_t2, "-", label="u2")
    plt.legend(loc="best")
    plt.tight_layout(pad=0.1)
    plt.show()
