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
nu = 0.003  # Mass ratio
eta_0 = 0.0  # Initial eta
mu = 0.6  # Coefficient of friction

# Import constants
gamma = 1 / (1 + nu)
k_0 = 1  # moment of inertia about centre
k_g = 1 - gamma**2  # moment of inertia about centre of gravity
theta_i = 0  # Initial theta
theta_f = 2 * pi  # Final theta
N = 700  # Number of divisions.
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


def Xi_initial():
    return sqrt(eta_0)


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
    return sqrt(eta_roll(x, eta, theta3))


# Spinning


def S_spin(x):
    return gamma * sin(x) - mu * (1 + gamma * cos(x))


def dn_dx_spin(x, eta):
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


def dXi_dt_spin(x, Xi, solution_y):
    func = solution_y.y[0, np.argmin(np.abs(solution_y.t - x))]
    return (
        sin(phi)
        - gamma * (zeta_spin(x, func) * cos(x) - func * sin(x))
        - mu * N_spin(x, func)
    ) / sqrt(((func)))


# Skidding


def S_skid(x):
    return gamma * sin(x) + mu * (1 + gamma * cos(x))


def zeta_skid(x, eta):
    return (
        (gamma * sin(x) + mu * (1 + gamma * cos(x)))
        * (cos(phi) - gamma * eta * cos(x))
        / (k_g + (gamma * sin(x) + mu * (1 + gamma * cos(x))) * gamma * sin(x))
    )


def dn_dx_skid(x, eta):
    return 2 * zeta_skid(x, eta)


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


def eta(x, func, theta_start, theta_end, thetaRange, eta_i):
    solskid = solve_ivp(
        func,
        (theta_start, theta_end),
        np.array([eta_i]),
        t_eval=thetaRange,
    )
    return solskid


def dXi_dt_skid(x, Xi, solution_y):
    func = solution_y.y[0, np.argmin(np.abs(solution_y.t - x))]
    return (
        sin(phi)
        - gamma * (zeta_skid(x, func) * cos(x) - func * sin(x))
        - mu * N_skid(x, func)
    ) / sqrt(((func)))


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
Xi_final[0] = Xi_initial()
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
        x_values_III = nu
        y_values_III = eta_0
        break
    elif Xi_final[i - 1] < 0:
        print("Roll back")
        x_values_III = nu
        y_values_III = eta_0
        break
        pass
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
                etaNew = eta_final[i - 1]
                thetaNew = linspace(theta[i - 1], theta_f, num=(N - i))
                solskid = eta(
                    thetaNew[0], dn_dx_skid, theta[i - 1], theta_f, thetaNew, etaNew
                )
                solvel_skid = solve_ivp(
                    lambda x, Xi: dXi_dt_skid(x, Xi, solskid),
                    (theta[i - 1], theta_f),
                    Xi_initial,
                    t_eval=thetaNew,
                )
                eta_final[i] = solskid.y[0, 0]
                Xi_final[i] = solvel_skid.y[0, 0]
                if eta_final[i] < 0 or Xi_final[i] < 0:
                    print("Roll back")
                    break
                root_eta[i] = sqrt(eta_final[i])
                Xi_initial = np.array([Xi_final[i - 1]])
                zeta_final[i] = zeta_skid(thetaNew[0], eta_final[i])
                N_final[i] = N_skid(thetaNew[0], eta_final[i])
                F_final[i] = -mu * N_final[i]
                F_N_final[i] = -mu
                print("skid")
            elif skid == True:
                roll = False
                spin = False
                sk = sk + 1
                eta_final[i] = solskid.y[0, sk]
                Xi_final[i] = solvel_skid.y[0, sk]
                if eta_final[i] < 0 or Xi_final[i] < 0:
                    print("Roll back")
                    break
                root_eta[i] = sqrt(eta_final[i])
                zeta_final[i] = zeta_skid(thetaNew[sk], eta_final[i])
                N_final[i] = N_skid(thetaNew[sk], eta_final[i])
                F_final[i] = -mu * N_final[i]
                F_N_final[i] = -mu
                print("skid")
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
                etaNew = eta_final[i - 1]

                thetaNew = linspace(theta[i - 1], theta_f, num=(N - i))
                solspin = eta(
                    thetaNew[0], dn_dx_spin, theta[i - 1], theta_f, thetaNew, etaNew
                )
                Xi_initial = np.array([Xi_final[i - 1]])
                solvel_spin = solve_ivp(
                    lambda x, Xi: dXi_dt_spin(x, Xi, solspin),
                    (theta[i - 1], theta_f),
                    Xi_initial,
                    t_eval=thetaNew,
                )
                eta_final[i] = solspin.y[0, 0]
                Xi_final[i] = solvel_spin.y[0, 0]
                if eta_final[i] < 0 or Xi_final[i] < 0:
                    print("Roll back")
                    break
                root_eta[i] = sqrt(eta_final[i])
                zeta_final[i] = zeta_spin(thetaNew[0], eta_final[i])
                N_final[i] = N_spin(thetaNew[0], eta_final[i])
                F_final[i] = mu * N_final[i]
                F_N_final[i] = mu
                print("spin")
            elif spin == True:
                roll = False
                skid = False
                sp = sp + 1
                eta_final[i] = solspin.y[0, sp]
                Xi_final[i] = solvel_spin.y[0, sp]
                if eta_final[i] < 0 or Xi_final[i] < 0:
                    print("Roll back")
                    break
                root_eta[i] = sqrt(eta_final[i])
                zeta_final[i] = zeta_spin(thetaNew[sp], eta_final[i])
                N_final[i] = N_spin(thetaNew[sp], eta_final[i])
                F_final[i] = mu * N_final[i]
                F_N_final[i] = mu
                print("spin")
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
            if eta_final[i] < 0 or Xi_final[i] < 0:
                print("Roll back")
                break
            root_eta[i] = sqrt(eta_final[i])
            F_final[i] = F_roll(theta[i], etaNew, theta_start)
            N_final[i] = N_roll(theta[i], etaNew, theta_start)
            zeta_final[i] = zeta_roll(theta[i], etaNew, theta_start)
            F_N_final[i] = F_N_roll(theta[i], etaNew, theta_start)
            print("roll")
        elif roll == True:
            skid = False
            spin = False
            eta_final[i] = eta_roll(theta[i], etaNew, theta_start)
            Xi_final[i] = Xi_roll(theta[i], etaNew, theta_start)
            if eta_final[i] < 0 or Xi_final[i] < 0:
                print("Roll back")
                break
            root_eta[i] = sqrt(eta_final[i])
            F_final[i] = F_roll(theta[i], etaNew, theta_start)
            N_final[i] = N_roll(theta[i], etaNew, theta_start)
            zeta_final[i] = zeta_roll(theta[i], etaNew, theta_start)
            F_N_final[i] = F_N_roll(theta[i], etaNew, theta_start)
            print("roll")

# print(x_values_III)
# print(y_values_III)
plt.plot(theta, root_eta, label="root eta")
plt.plot(theta, eta_final, label="eta")
plt.plot(theta, N_final, label="N")
plt.plot(theta, F_N_final, label="F/N")
plt.plot(theta, Xi_final, label="Velocity")
# plt.plot(theta, zeta_final, label="zeta")
plt.legend(loc="upper left")
plt.show()
