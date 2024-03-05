from numpy import (
    sin,
    cos,
    sqrt,
    linspace,
)
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

pi = 3.1415926535897932

# Parameters of the simulation
phi = 0  # Angle of slope
nu = 0.3  # Mass ratio
eta_0 = 0.5  # Initial eta
mu = 0.6  # Coefficient of friction

# Import constants
gamma = 1 / (1 + nu)
k_0 = 1  # moment of inertia about centre
k_g = 1 - gamma**2  # moment of inertia about centre of gravity
theta_i = 0  # Initial theta
theta_f = 2 * pi  # Final theta
N = 50000  # Number of divisions.
theta = linspace(theta_i, theta_f, num=N)
crit_velocity = 1 / gamma * cos(phi)
crit_friction = 0.5 * (1 - gamma) * sin(phi) / (gamma * (crit_velocity - eta_0))


# Moment of inertia about instantaneous centre of rotation
def k_c(x):
    return 2 * (1 + gamma * cos(x))


# The initial values


def F_initial():
    return -0.5 * (1 - gamma) * sin(phi)


def N_initial():
    return gamma * (crit_velocity - eta_0)


def F_N_initial():
    return F_initial() / N_initial()


def eta_initial():
    return eta_0


def zeta_initial():
    return sin(phi) / 2


# Pure rolling
def C(eta, theta3):
    return (
        0.5 * eta * (k_0 + 1 + 2 * gamma * cos(theta3))
        + gamma * cos(phi + theta3)
        - theta3 * sin(phi)
    )


def eta_roll(x, eta, theta3):
    return (
        2
        * (-gamma * cos(x + phi) + x * sin(phi) + C(eta, theta3))
        / (2 * (1 + gamma * cos(x)))
    )


def zeta_roll(x, eta, theta3):
    return (
        1
        / (2 * (1 + gamma * cos(x)))
        * (
            gamma
            * (
                2
                * (-gamma * cos(x + phi) + x * sin(phi) + C(eta, theta3))
                / (2 * (1 + gamma * cos(x)))
            )
            * sin(x)
            + gamma * sin(x + phi)
            + sin(phi)
        )
    )


def F_roll(x, eta, theta3):
    return (
        (1 + gamma * cos(x))
        * (
            1
            / (2 * (1 + gamma * cos(x)))
            * (
                gamma
                * 2
                * (-gamma * cos(x + phi) + x * sin(phi) + C(eta, theta3))
                / (2 * (1 + gamma * cos(x)))
                * sin(x)
                + gamma * sin(x + phi)
                + sin(phi)
            )
        )
        - gamma
        * (
            2
            * (-gamma * cos(x + phi) + x * sin(phi) + C(eta, theta3))
            / (2 * (1 + gamma * cos(x)))
        )
        * sin(x)
        - sin(phi)
    )


def N_roll(x, eta, theta3):
    return cos(phi) - gamma * (
        (
            1
            / (2 * (1 + gamma * cos(x)))
            * (
                gamma
                * (
                    2
                    * (-gamma * cos(x + phi) + x * sin(phi) + C(eta, theta3))
                    / (2 * (1 + gamma * cos(x)))
                )
                * sin(x)
                + gamma * sin(x + phi)
                + sin(phi)
            )
        )
        * sin(x)
        + 2
        * (-gamma * cos(x + phi) + x * sin(phi) + C(eta, theta3))
        / (2 * (1 + gamma * cos(x)))
        * cos(x)
    )


def F_N_roll(x, eta, theta3):
    return (
        (
            (1 + gamma * cos(x))
            * (
                1
                / (2 * (1 + gamma * cos(x)))
                * (
                    gamma
                    * 2
                    * (-gamma * cos(x + phi) + x * sin(phi) + C(eta, theta3))
                    / (2 * (1 + gamma * cos(x)))
                    * sin(x)
                    + gamma * sin(x + phi)
                    + sin(phi)
                )
            )
            - gamma
            * (
                2
                * (-gamma * cos(x + phi) + x * sin(phi) + C(eta, theta3))
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
                        * (-gamma * cos(x + phi) + x * sin(phi) + C(eta, theta3))
                        / (2 * (1 + gamma * cos(x)))
                    )
                    * sin(x)
                    + gamma * sin(x + phi)
                    + sin(phi)
                )
            )
            * sin(x)
            + 2
            * (-gamma * cos(x + phi) + x * sin(phi) + C(eta, theta3))
            / (2 * (1 + gamma * cos(x)))
            * cos(x)
        )
    )


def Xi_roll(x, eta, theta3):
    return sqrt(np.abs(eta_roll(x, eta, theta3)))


# Spinning


def S_spin(x):
    return gamma * sin(x) - mu * (1 + gamma * cos(x))


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


def derivatives_spin(x, z):
    eta, Xi = z
    dndx = (
        2
        * (gamma * sin(x) - mu * (1 + gamma * cos(x)))
        * (cos(phi) - gamma * eta * cos(x))
        / (k_g + (gamma * sin(x) - mu * (1 + gamma * cos(x))) * gamma * sin(x))
    )
    dXidx = (
        sin(phi)
        - gamma * (zeta_spin(x, eta) * cos(x) - eta * sin(x))
        - mu * N_spin(x, eta)
    ) / sqrt(np.abs((eta)))
    return dndx, dXidx


# Skidding


def S_skid(x):
    return gamma * sin(x) + mu * (1 + gamma * cos(x))


def zeta_skid(x, eta):
    return (
        (gamma * sin(x) + mu * (1 + gamma * cos(x)))
        * (cos(phi) - gamma * eta * cos(x))
        / (k_g + (gamma * sin(x) + mu * (1 + gamma * cos(x))) * gamma * sin(x))
    )


def N_skid(x, eta):
    return cos(phi) - gamma * (
        (
            (
                (gamma * sin(x) + mu * (1 + gamma * cos(x)))
                * (cos(phi) - gamma * eta * cos(x))
                / (k_g + (gamma * sin(x) + mu * (1 + gamma * cos(x))) * gamma * sin(x))
            )
        )
        * sin(x)
        + eta * cos(x)
    )


def F_skid(N):
    return -mu * N


def derivatives_skid(x, z):
    eta, Xi, E = z
    dndx = 2 * (
        (
            (gamma * sin(x) + mu * (1 + gamma * cos(x)))
            * (cos(phi) - gamma * eta * cos(x))
            / (k_g + (gamma * sin(x) + mu * (1 + gamma * cos(x))) * gamma * sin(x))
        )
    )
    dXidx = (
        sin(phi)
        - gamma * (zeta_skid(x, eta) * cos(x) - eta * sin(x))
        - mu * N_skid(x, eta)
    ) / sqrt(np.abs((eta)))
    dEdx = np.abs(mu * N_skid(x, eta) * Xi / (sqrt(np.abs(eta))))
    return dndx, dXidx, dEdx


def energy_lost(x, E, force, solution_eta, solution_velocity):
    eta = solution_eta.y[0, np.argmin(np.abs(solution_eta.t - x))]
    velocity = solution_velocity.y[0, np.argmin(np.abs(solution_velocity.t - x))]
    dEdx = force(x, eta) * velocity / (sqrt(np.abs((eta))))
    return dEdx


eta_final = np.zeros(N)
F_final = np.zeros(N)
N_final = np.zeros(N)
F_N_final = np.zeros(N)
zeta_final = np.zeros(N)
Xi_final = np.zeros(N)
a_final = np.zeros(N)
root_eta = np.zeros(N)
theta_start = theta[0]
events_theta = []
roll = False
spin = False
skid = False
sp = 0
sk = 0
eta_final[0] = eta_initial()
F_final[0] = F_initial()
N_final[0] = N_initial()
zeta_final[0] = zeta_initial()
F_N_final[0] = F_N_initial()
Xi_final[0] = sqrt(eta_final[0])
root_eta[0] = sqrt(eta_final[0])

if eta_0 > crit_velocity:
    print("Immediate Hop")
elif eta_0 < crit_velocity and mu > crit_friction:
    roll = True
    spin = False
    skid = False
    etaNew = eta_0
for i in range(1, N):
    if N_final[i - 1] < 0 and eta_final[i - 1] * cos(theta[i - 1]) > crit_velocity:
        print("Hop")
        break
    elif abs(F_N_final[i - 1]) >= mu:
        # Skid
        if F_final[i - 1] < 0:
            F_N_final[i - 1] = -mu
            if skid == False:
                events_theta.append(theta[i - 1])
                sk = 0
                skid = True
                roll = False
                spin = False
                thetaNew = theta[i:]
                initial_values = np.array([eta_final[i - 1], Xi_final[i - 1], 0])
                solution_skid = solve_ivp(
                    derivatives_skid,
                    (theta[i - 1], theta_f),
                    initial_values,
                    method="Radau",
                    t_eval=thetaNew,
                )
                eta_final[i] = solution_skid.y[0, 0]
                Xi_final[i] = solution_skid.y[1, 0]
                root_eta[i] = sqrt(eta_final[i])
                zeta_final[i] = zeta_skid(thetaNew[0], eta_final[i])
                N_final[i] = N_skid(thetaNew[0], eta_final[i])
                F_final[i] = -mu * N_final[i]
                F_N_final[i] = -mu
            elif skid == True:
                roll = False
                spin = False
                sk = sk + 1
                eta_final[i] = solution_skid.y[0, sk]
                Xi_final[i] = solution_skid.y[1, sk]
                if eta_final[i] < 0:
                    print("Roll back")
                    break
                root_eta[i] = sqrt(eta_final[i])
                zeta_final[i] = zeta_skid(thetaNew[sk], eta_final[i])
                N_final[i] = N_skid(thetaNew[sk], eta_final[i])
                F_final[i] = -mu * N_final[i]
                F_N_final[i] = -mu
                if Xi_final[i] < root_eta[i]:
                    skid = False
                    roll = False
                    spin = False
                    F_N_final[i] = -mu + 0.00001
        # Spin
        elif F_final[i - 1] > 0:
            F_N_final[i - 1] = mu
            if spin == False:
                events_theta.append(theta[i - 1])
                sp = 0
                spin = True
                roll = False
                skid = False
                thetaNew = theta[i:]
                initial_values = np.array([eta_final[i - 1], Xi_final[i - 1]])
                solution_spin = solve_ivp(
                    derivatives_spin,
                    (theta[i - 1], theta_f),
                    initial_values,
                    method="Radau",
                    t_eval=thetaNew,
                )
                eta_final[i] = solution_spin.y[0, 0]
                Xi_final[i] = solution_spin.y[1, 0]
                root_eta[i] = sqrt(eta_final[i])
                zeta_final[i] = zeta_spin(thetaNew[0], eta_final[i])
                N_final[i] = N_spin(thetaNew[0], eta_final[i])
                F_final[i] = mu * N_final[i]
                F_N_final[i] = mu
            elif spin == True:
                roll = False
                skid = False
                sp = sp + 1
                eta_final[i] = solution_spin.y[0, sp]
                Xi_final[i] = solution_spin.y[1, sp]
                if eta_final[i] < 0:
                    print("Roll back")
                    break
                root_eta[i] = sqrt(eta_final[i])
                zeta_final[i] = zeta_spin(thetaNew[sp], eta_final[i])
                N_final[i] = N_spin(thetaNew[sp], eta_final[i])
                F_final[i] = mu * N_final[i]
                F_N_final[i] = mu
                if Xi_final[i] > root_eta[i]:
                    skid = False
                    roll = False
                    spin = False
                    F_N_final[i] = mu - 0.00001
    else:
        if roll == False:
            roll = True
            skid = False
            spin = False
            etaNew = eta_final[i - 1]
            theta_start = theta[i]
            eta_final[i] = eta_roll(theta[i], etaNew, theta_start)
            Xi_final[i] = Xi_roll(theta[i], etaNew, theta_start)
            if eta_final[i] < 0:
                print("Roll back")
                break
            root_eta[i] = sqrt(eta_final[i])
            F_final[i] = F_roll(theta[i], etaNew, theta_start)
            N_final[i] = N_roll(theta[i], etaNew, theta_start)
            zeta_final[i] = zeta_roll(theta[i], etaNew, theta_start)
            F_N_final[i] = F_N_roll(theta[i], etaNew, theta_start)
        elif roll == True:
            skid = False
            spin = False
            eta_final[i] = eta_roll(theta[i], etaNew, theta_start)
            Xi_final[i] = Xi_roll(theta[i], etaNew, theta_start)
            if eta_final[i] < 0:
                print("Roll back")
                break
            root_eta[i] = sqrt(eta_final[i])
            F_final[i] = F_roll(theta[i], etaNew, theta_start)
            N_final[i] = N_roll(theta[i], etaNew, theta_start)
            zeta_final[i] = zeta_roll(theta[i], etaNew, theta_start)
            F_N_final[i] = F_N_roll(theta[i], etaNew, theta_start)

# print(x_values_III)
# print(y_values_III)
# plt.plot(theta, root_eta, label="root eta")
plt.plot(theta, eta_final, label="eta")
plt.plot(theta, N_final, label="N")
plt.plot(theta, F_final)
plt.plot(theta, F_N_final, label="F/N")
plt.plot(theta, Xi_final, label="Velocity")
# plt.plot(theta, zeta_final, label="zeta")
plt.plot(solution_skid.t, solution_skid.y[2])
plt.legend(loc="upper left")
plt.show()
