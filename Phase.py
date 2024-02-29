from numpy import (
    sin,
    cos,
    sqrt,
    linspace,
)
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import multiprocessing
from itertools import product
import time
from datetime import datetime

start_time = time.time()

pi = 3.1415926535897932

# Parameters of the simulation
phi = 0.17  # Angle of slope
mu = 0.6  # Coefficient of friction
theta_iPhase = 0  # Initial theta
theta_fPhase = pi * 2  # Final theta
thetaPhase = linspace(theta_iPhase, theta_fPhase, num=400)
k_0 = 1  # moment of inertia about centre


def C(eta, gamma_variable):
    return 0.5 * eta * (k_0 + 1 + 2 * gamma_variable) + gamma_variable * cos(phi)


def F_N_Phase(x, eta, gamma_variable):
    return (
        (
            (1 + gamma_variable * cos(x))
            * (
                1
                / (2 * (1 + gamma_variable * cos(x)))
                * (
                    gamma_variable
                    * 2
                    * (
                        -gamma_variable * cos(x + phi)
                        + x * sin(phi)
                        + C(eta, gamma_variable)
                    )
                    / (2 * (1 + gamma_variable * cos(x)))
                    * sin(x)
                    + gamma_variable * sin(x + phi)
                    + sin(phi)
                )
            )
            - gamma_variable
            * (
                2
                * (
                    -gamma_variable * cos(x + phi)
                    + x * sin(phi)
                    + C(eta, gamma_variable)
                )
                / (2 * (1 + gamma_variable * cos(x)))
            )
            * sin(x)
            - sin(phi)
        )
    ) / (
        cos(phi)
        - gamma_variable
        * (
            (
                1
                / (2 * (1 + gamma_variable * cos(x)))
                * (
                    gamma_variable
                    * (
                        2
                        * (
                            -gamma_variable * cos(x + phi)
                            + x * sin(phi)
                            + C(eta, gamma_variable)
                        )
                        / (2 * (1 + gamma_variable * cos(x)))
                    )
                    * sin(x)
                    + gamma_variable * sin(x + phi)
                    + sin(phi)
                )
            )
            * sin(x)
            + 2
            * (-gamma_variable * cos(x + phi) + x * sin(phi) + C(eta, gamma_variable))
            / (2 * (1 + gamma_variable * cos(x)))
            * cos(x)
        )
    )


def x_parallel(phi, nu, eta_0, mu, N):

    # Import constants
    gamma = 1 / (1 + nu)
    k_0 = 1  # moment of inertia about centre
    k_g = 1 - gamma**2  # moment of inertia about centre of gravity
    theta_i = 0  # Initial theta
    theta_f = 2 * pi  # Final theta
    theta = linspace(theta_i, theta_f, num=N)
    crit_velocity = 1 / gamma * cos(phi)

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
        return sqrt(abs(eta_roll(x, eta, theta3)))

    # Spinning

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
                    / (
                        k_g
                        + (gamma * sin(x) - mu * (1 + gamma * cos(x))) * gamma * sin(x)
                    )
                )
            )
            * sin(x)
            + cos(x) * eta
        )

    def dXi_dt_spin(x, Xi, solution_y):
        func = solution_y.y[0, np.argmin(np.abs(solution_y.t - x))]
        return (
            sin(phi)
            - gamma * (zeta_spin(x, func) * cos(x) - func * sin(x))
            - mu * N_spin(x, func)
        ) / sqrt(abs((func)))

    # Skidding

    def zeta_skid(x, eta):
        return (
            (gamma * sin(x) + mu * (1 + gamma * cos(x)))
            * (cos(phi) - gamma * eta * cos(x))
            / (
                k_g
                + (gamma * sin(x) + mu * (1 + gamma * cos(x))) * gamma * sin(x)
                + 0.0000001
            )
        )

    def dn_dx_skid(x, eta):
        return 2 * (
            (
                (gamma * sin(x) + mu * (1 + gamma * cos(x)))
                * (cos(phi) - gamma * eta * cos(x))
                / (
                    k_g
                    + (gamma * sin(x) + mu * (1 + gamma * cos(x))) * gamma * sin(x)
                    + 0.0000001
                )
            )
        )

    def N_skid(x, eta):
        return cos(phi) - gamma * (
            (
                (
                    (gamma * sin(x) + mu * (1 + gamma * cos(x)))
                    * (cos(phi) - gamma * eta * cos(x))
                    / (
                        k_g
                        + (gamma * sin(x) + mu * (1 + gamma * cos(x))) * gamma * sin(x)
                        + 0.000001
                    )
                )
            )
            * sin(x)
            + eta * cos(x)
        )

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
        ) / sqrt(abs((func)))

    eta_final = np.zeros(N)
    F_final = np.zeros(N)
    N_final = np.zeros(N)
    F_N_final = np.zeros(N)
    Xi_final = np.zeros(N)
    root_eta = np.zeros(N)
    theta_start = theta[0]
    roll = False
    spin = False
    skid = False
    sp = 0
    sk = 0
    x_values_III = None
    y_values_III = None

    eta_final[0] = eta_0
    F_final[0] = -0.5 * (1 - gamma) * sin(phi)
    N_final[0] = gamma * (crit_velocity - eta_0)
    F_N_final[0] = F_final[0] / N_final[0]
    Xi_final[0] = sqrt(eta_0)
    root_eta[0] = sqrt(eta_final[0])

    if eta_0 <= crit_velocity:
        roll = True
        spin = False
        skid = False
        etaNew = eta_0
    for i in range(1, N):
        if eta_final[i - 1] < 0 or Xi_final[i] < 0:
            break
        elif (
            N_final[i - 1] <= 0 and eta_final[i - 1] * cos(theta[i - 1]) > crit_velocity
        ):
            x_values_III = nu
            y_values_III = eta_0
            break
        elif abs(F_N_final[i - 1]) >= mu:
            # Skid
            if F_final[i - 1] < 0:
                F_N_final[i - 1] = -mu
                if skid == False:
                    sk = 0
                    sp = 0
                    skid = True
                    roll = False
                    spin = False
                    etaNew = eta_final[i - 1]
                    thetaNew = linspace(theta[i - 1], theta_f, num=(N - i))
                    solskid = eta(
                        thetaNew[0],
                        dn_dx_skid,
                        theta[i - 1],
                        theta_f,
                        thetaNew,
                        etaNew,
                    )
                    eta_final[i] = solskid.y[0, 0]
                    if eta_final[i] < 0 or Xi_final[i] < 0:
                        break
                    root_eta[i] = sqrt(eta_final[i])
                    Xi_initial = np.array([Xi_final[i - 1]])
                    solvel_skid = solve_ivp(
                        lambda x, Xi: dXi_dt_skid(x, Xi, solskid),
                        (theta[i - 1], theta_f),
                        Xi_initial,
                        t_eval=thetaNew,
                    )
                    Xi_final[i] = solvel_skid.y[0, 0]
                    N_final[i] = N_skid(thetaNew[0], eta_final[i])
                    F_final[i] = -mu * N_final[i]
                    F_N_final[i] = -mu
                elif skid == True:
                    roll = False
                    spin = False
                    sk = sk + 1
                    eta_final[i] = solskid.y[0, sk]
                    if eta_final[i] < 0 or Xi_final[i] < 0:
                        break
                    root_eta[i] = sqrt(eta_final[i])
                    Xi_final[i] = solvel_skid.y[0, sk]
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
                    sp = 0
                    sk = 0
                    spin = True
                    roll = False
                    skid = False
                    etaNew = eta_final[i - 1]
                    thetaNew = linspace(theta[i - 1], theta_f, num=(N - i))
                    solspin = eta(
                        thetaNew[0],
                        dn_dx_spin,
                        theta[i - 1],
                        theta_f,
                        thetaNew,
                        etaNew,
                    )
                    eta_final[i] = solspin.y[0, 0]
                    if eta_final[i] < 0 or Xi_final[i] < 0:
                        break
                    root_eta[i] = sqrt(eta_final[i])
                    Xi_initial = np.array([Xi_final[i - 1]])
                    solvel_spin = solve_ivp(
                        lambda x, Xi: dXi_dt_spin(x, Xi, solspin),
                        (theta[i - 1], theta_f),
                        Xi_initial,
                        t_eval=thetaNew,
                    )
                    Xi_final[i] = solvel_spin.y[0, 0]
                    N_final[i] = N_spin(thetaNew[0], eta_final[i])
                    F_final[i] = mu * N_final[i]
                    F_N_final[i] = mu
                elif spin == True:
                    roll = False
                    skid = False
                    sp = sp + 1
                    eta_final[i] = solspin.y[0, sp]
                    if eta_final[i] < 0 or Xi_final[i] < 0:
                        break
                    root_eta[i] = sqrt(eta_final[i])
                    Xi_final[i] = solvel_spin.y[0, sp]
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
                if eta_final[i] < 0 or Xi_final[i] < 0:
                    break
                root_eta[i] = sqrt(eta_final[i])
                F_final[i] = F_roll(theta[i], etaNew, theta_start)
                N_final[i] = N_roll(theta[i], etaNew, theta_start)
                F_N_final[i] = F_N_roll(theta[i], etaNew, theta_start)
                Xi_final[i] = Xi_roll(theta[i], etaNew, theta_start)
            elif roll == True:
                skid = False
                spin = False
                eta_final[i] = eta_roll(theta[i], etaNew, theta_start)
                if eta_final[i] < 0 or Xi_final[i] < 0:
                    break
                root_eta[i] = sqrt(eta_final[i])
                F_final[i] = F_roll(theta[i], etaNew, theta_start)
                N_final[i] = N_roll(theta[i], etaNew, theta_start)
                F_N_final[i] = F_N_roll(theta[i], etaNew, theta_start)
                Xi_final[i] = Xi_roll(theta[i], etaNew, theta_start)
    return x_values_III, y_values_III


Region_III_mass = []
Region_III_vel = []
Region_II_mass = []
Region_II_vel = []
Region_IV_mass = []
Region_IV_vel = []
n = 10  # Mass divisions
m = 20  # Velocity divisions
N = 30000  # Angle divisions
n_III = n
m_III = m
mass_ratio = linspace(0.06, 0.15, n)
initial_velocity = linspace(0.000000000000001, 0.3, m)
for i in range(n):
    for j in range(m):
        eta = initial_velocity[j]
        gamma = (mass_ratio[i] + 1) ** (-1)
        if (
            abs(max(F_N_Phase(thetaPhase, eta, gamma))) >= mu
            or abs(min(F_N_Phase(thetaPhase, eta, gamma))) >= mu
        ):
            Region_II_mass.append(mass_ratio[i])
            Region_II_vel.append(eta)
            break


def crit_velocity(mass_ratio):
    return (1 + mass_ratio) * cos(phi)


Region_IV_mass = mass_ratio
Region_IV_vel = crit_velocity(mass_ratio)


def simulate_parallel(params):
    mass_ratio, initial_velocity = params
    a, b = x_parallel(phi, mass_ratio, initial_velocity, mu, N)
    return a, b


if __name__ == "__main__":

    # Use multiprocessing Pool for parallelization
    with multiprocessing.Pool() as pool:
        # Parallelize simulation for different parameter sets
        params_list = product(mass_ratio, initial_velocity)
        results = pool.map(simulate_parallel, params_list)

    # Extract results until the first valid result is found for each mass ratio
    Region_III_mass = []
    Region_III_vel = []
    found_for_ratio = set()  # Keep track of mass ratios for which a result is found

    for result in results:
        mass_ratio, initial_velocity = result
        if mass_ratio not in found_for_ratio and initial_velocity is not None:
            Region_III_mass.append(mass_ratio)
            Region_III_vel.append(initial_velocity)
            found_for_ratio.add(mass_ratio)


# Record the end time
end_time = time.time()

# Calculate the runtime
runtime = end_time - start_time

print(f"Runtime: {runtime} seconds")

# Open a file for writing
with open("III.0.6.0.17.txt", "a") as file:
    file.write("X Divisions: " + str(n_III) + "\n")
    file.write("Y Divisions: " + str(m_III) + "\n")
    file.write("Angle Divisions: " + str(N) + "\n")
    for i in range(n):
        file.write(str(Region_III_mass[i]) + "," + str(Region_III_vel[i]) + "\n")
    file.write("\n")
file.close()

print(Region_IV_mass)
print(Region_IV_vel)
print(Region_II_mass)
print(Region_II_vel)
print(Region_III_mass)
print(Region_III_vel)
plt.plot(Region_II_mass, Region_II_vel)
plt.plot(Region_IV_mass, Region_IV_vel)
plt.plot(Region_III_mass, Region_III_vel)
plt.ylim(0, 2)
plt.xlim(0, 1)
current_time = datetime.now()
plotname = (
    str(n_III)
    + " "
    + str(m_III)
    + " "
    + str(N)
    + " "
    + str(mu)
    + " "
    + str(phi)
    + str(current_time)
    + ".png"
)
# plt.savefig(plotname, format="png")
plt.show()
