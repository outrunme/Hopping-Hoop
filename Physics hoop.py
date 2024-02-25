from numpy import (
    sin,
    cos,
    sqrt,
    radians,
    degrees,
    sign,
    absolute,
    diff,
    linspace,
    concatenate,
    isclose,
    arange,
    interp,
)
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import animation

pi = 3.1415926535897932

# Parameters of the simulation
phi = 0  # Angle of slope
nu = 0.5  # Mass ratio
eta_0 = np.array([1.1])  # Initial eta
mu = 0.6  # Coefficient of friction

# Import constants
gamma = 1 / (1 + nu)
k_0 = 1  # moment of inertia about centre
k_g = 1 - gamma**2  # moment of inertia about centre of gravity
theta_i = 0  # Initial theta
theta_f = 2 * pi  # Final theta
theta = linspace(theta_i, theta_f, num=360)
crit_velocity = 1 / gamma * cos(phi)
abserr = 1e-8
relerr = 1e-8


# Moment of inertia about instantaneous centre of rotation
def k_c(x):
    return 2 * (1 + gamma * cos(x))


# Pure rolling
C = 0.5 * eta_0 * (k_0 + 1 + 2 * gamma) + gamma * cos(phi)


def eta(x):
    return 2 * (-gamma * cos(x + phi) + x * sin(phi) + C) / (2 * (1 + gamma * cos(x)))


def zeta(x):
    return (
        1
        / (2 * (1 + gamma * cos(x)))
        * (
            gamma
            * (
                2
                * (-gamma * cos(x + phi) + x * sin(phi) + C)
                / (2 * (1 + gamma * cos(x)))
            )
            * sin(x)
            + gamma * sin(x + phi)
            + sin(phi)
        )
    )


def F(x):
    return (
        (1 + gamma * cos(x))
        * (
            1
            / (2 * (1 + gamma * cos(x)))
            * (
                gamma
                * 2
                * (-gamma * cos(x + phi) + x * sin(phi) + C)
                / (2 * (1 + gamma * cos(x)))
                * sin(x)
                + gamma * sin(x + phi)
                + sin(phi)
            )
        )
        - gamma
        * (2 * (-gamma * cos(x + phi) + x * sin(phi) + C) / (2 * (1 + gamma * cos(x))))
        * sin(x)
        - sin(phi)
    )


def N(x):
    return cos(phi) - gamma * (
        (
            1
            / (2 * (1 + gamma * cos(x)))
            * (
                gamma
                * (
                    2
                    * (-gamma * cos(x + phi) + x * sin(phi) + C)
                    / (2 * (1 + gamma * cos(x)))
                )
                * sin(x)
                + gamma * sin(x + phi)
                + sin(phi)
            )
        )
        * sin(x)
        + 2
        * (-gamma * cos(x + phi) + x * sin(phi) + C)
        / (2 * (1 + gamma * cos(x)))
        * cos(x)
    )


def F_N(x):
    return (
        (
            (1 + gamma * cos(x))
            * (
                1
                / (2 * (1 + gamma * cos(x)))
                * (
                    gamma
                    * 2
                    * (
                        -gamma * cos(x + phi)
                        + x * sin(phi)
                        + (0.5 * eta_0 * (k_0 + 1 + 2 * gamma) + gamma * cos(phi))
                    )
                    / (2 * (1 + gamma * cos(x)))
                    * sin(x)
                    + gamma * sin(x + phi)
                    + sin(phi)
                )
            )
            - gamma
            * (
                2
                * (
                    -gamma * cos(x + phi)
                    + x * sin(phi)
                    + (0.5 * eta_0 * (k_0 + 1 + 2 * gamma) + gamma * cos(phi))
                )
                / (2 * (1 + gamma * cos(x)))
            )
            * sin(x)
            - sin(phi)
        )
    ) / (
        cos(phi)
        - gamma
        * (
            (
                1
                / (2 * (1 + gamma * cos(x)))
                * (
                    gamma
                    * (
                        2
                        * (
                            -gamma * cos(x + phi)
                            + x * sin(phi)
                            + (0.5 * eta_0 * (k_0 + 1 + 2 * gamma) + gamma * cos(phi))
                        )
                        / (2 * (1 + gamma * cos(x)))
                    )
                    * sin(x)
                    + gamma * sin(x + phi)
                    + sin(phi)
                )
            )
            * sin(x)
            + 2
            * (
                -gamma * cos(x + phi)
                + x * sin(phi)
                + (0.5 * eta_0 * (k_0 + 1 + 2 * gamma) + gamma * cos(phi))
            )
            / (2 * (1 + gamma * cos(x)))
            * cos(x)
        )
    )


# Spinning


def S_spin(x):
    return gamma * sin(x) - mu * (1 + gamma * cos(x))


def dn_dx_spin(eta, x):
    return (
        2
        * (gamma * sin(x) - mu * (1 + gamma * cos(x)))
        * (cos(phi) - gamma * eta * cos(x))
        / (k_g + (gamma * sin(x) - mu * (1 + gamma * cos(x))) * gamma * sin(x))
    )


def zeta_spin(x, eta):
    return (
        (gamma * sin(x) - mu * (1 + gamma * cos(x)))
        * (cos(phi) - gamma * eta * cos(x))
        / (k_g + (gamma * sin(x) - mu * (1 + gamma * cos(x))) * gamma * sin(x))
    )


def N_spin(x, eta):
    return cos(phi) - gamma * (
        (
            (
                (gamma * sin(x) - mu * (1 + gamma * cos(x)))
                * (cos(phi) - gamma * eta * cos(x))
                / (k_g + (gamma * sin(x) - mu * (1 + gamma * cos(x))) * gamma * sin(x))
            )
        )
        * sin(x)
        + cos(x) * eta
    )


def F_spin(N):
    return mu * N


# Skidding


def S_skid(x):
    return gamma * sin(x) + mu * (1 + gamma * cos(x))


def dn_dx_skid(eta, x):
    return (
        2
        * (gamma * sin(x) + mu * (1 + gamma * cos(x)))
        * (cos(phi) - gamma * eta * cos(x))
        / (k_g + (gamma * sin(x) + mu * (1 + gamma * cos(x))) * gamma * sin(x))
    )


def zeta_spin(x, eta):
    return (
        (gamma * sin(x) + mu * (1 + gamma * cos(x)))
        * (cos(phi) - gamma * eta * cos(x))
        / (k_g + (gamma * sin(x) + mu * (1 + gamma * cos(x))) * gamma * sin(x))
    )


def N_spin(x, eta):
    return cos(phi) - gamma * (
        (
            (
                (gamma * sin(x) + mu * (1 + gamma * cos(x)))
                * (cos(phi) - gamma * eta * cos(x))
                / (k_g + (gamma * sin(x) + mu * (1 + gamma * cos(x))) * gamma * sin(x))
            )
        )
        * sin(x)
        + cos(x) * eta
    )


def F_spin(N):
    return -mu * N


# solspin = solve_ivp(dn_dx_spin, (theta_i, theta_f), eta_0, t_eval=theta)
# x_spin = solspin.t
# eta_spin = solspin.y[0]

# zeta_spin_ = zeta_spin(x_spin, eta_spin)
# N_spin_ = N_spin(x_spin, eta_spin)
# F_spin_ = F_spin(N_spin_)


# solskid = solve_ivp(dn_dx_skid, (theta_i, theta_f), eta_0, t_eval=theta)
# x_spin = solskid.t
# eta_spin = solskid.y[0]

# zeta_spin_ = zeta_spin(x_spin, eta_spin)
# N_spin_ = N_spin(x_spin, eta_spin)
# F_spin_ = F_spin(N_spin_)

eta_final = np.zeros(360)
F_final = np.zeros(360)
N_final = np.zeros(360)
F_N_final = np.zeros(360)
zeta_final = np.zeros(360)
events_theta = []


if eta_0 < crit_velocity:
    eta_final[0] = eta(theta[0])
    F_final[0] = F(theta[0])
    N_final[0] = N(theta[0])
    zeta_final[0] = zeta(theta[0])
    F_N_final[0] = F_N(theta[0])
    for i in range(1, 360):
        if abs(F_N_final[i - 1]) >= mu:
            events_theta.append(theta[i - 1])
            if F_final[i - 1] < 0:
                eta2 = np.array([eta_final[i - 1]])
                thetaNew = linspace(theta[i - 1], theta_f, num=(360 - i))
                solspin = solve_ivp(
                    dn_dx_spin,
                    (theta[i - 1], theta_f),
                    eta2,
                    t_eval=thetaNew,
                )
                eta_temp = solspin.y[0]
                eta_final[i] = eta_temp[0]
                zeta_final[i] = zeta_spin(thetaNew[0], eta_final[i])
                N_final[i] = N_spin(thetaNew[0], eta_final[i])
                F_final[i] = F_spin(N_final[i])
                F_N_final[i] = -mu

        else:
            eta_final[i] = eta(theta[i])
            F_final[i] = F(theta[i])
            N_final[i] = N(theta[i])
            zeta_final[i] = zeta(theta[i])
            F_N_final[i] = F_N(theta[i])

plt.plot(theta, eta_final, label="eta")
plt.plot(theta, F_final, label="F")
plt.plot(theta, N_final, label="N")
plt.plot(theta, F_N_final, label="F/N")
plt.plot(theta, zeta_final, label="zeta")
plt.legend(loc="upper left")
plt.show()


# if abs(max(F_N_temp)) < mu:
#     eta_final = eta(theta)
#     F_final = F(theta)
#     N_final = N(theta)
#     F_N_final = F_N(theta)
#     zeta_final = zeta(theta)
#     plt.plot(theta, eta_final, label="eta")
#     plt.plot(theta, F_final, label="F")
#     plt.plot(theta, N_final, label="N")
#     plt.plot(theta, F_N_final, label="F/N")
#     plt.plot(theta, zeta_final, label="zeta")
#     plt.legend(loc="upper left")
#     plt.show()
# else:
#     print("slips!")


# Rolling

# plt.plot(theta, eta_roll, label="eta")
# plt.plot(theta, F_roll, label="F")
# plt.plot(theta, N_roll, label="N")
# plt.plot(theta, F_N_roll, label="F/N")
# plt.plot(theta, zeta_roll, label="zeta")
# plt.legend(loc="upper left")
# plt.show()

# plt.plot(x_spin, eta_spin, label="eta")
# plt.plot(x_spin, zeta_spin_, label="zeta")
# plt.plot(x_spin, N_spin_, label="N")
# plt.plot(x_spin, F_spin_, label="F")
# plt.legend(loc="upper left")
# plt.show()

# plt.plot(x_spin, eta_spin, label="eta")
# plt.plot(x_spin, zeta_spin_, label="zeta")
# plt.plot(x_spin, N_spin_, label="N")
# plt.plot(x_spin, F_spin_, label="F")
# plt.legend(loc="upper left")
# plt.show()
