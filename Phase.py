from numpy import (
    sin,
    cos,
    sqrt,
    linspace,
)
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from multiprocessing import Queue, Manager, Pool
import time
from datetime import datetime

start_time = time.time()

pi = 3.1415926535897932

# Parameters of the simulation
phi = 0  # Angle of slope
mu = 0.6  # Coefficient of friction
theta_iPhase = 0  # Initial theta
theta_fPhase = pi * 2  # Final theta
thetaPhase = linspace(theta_iPhase, theta_fPhase, num=400)
k_0 = 1  # moment of inertia about centre


def crit_velocity(mass_ratio):
    return (1 + mass_ratio) * cos(phi)


def avg(upper_bound, lower_bound):
    return (upper_bound + lower_bound) / 2


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
                        + (
                            0.5 * eta * (k_0 + 1 + 2 * gamma_variable)
                            + gamma_variable * cos(phi)
                        )
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
                    + (
                        0.5 * eta * (k_0 + 1 + 2 * gamma_variable)
                        + gamma_variable * cos(phi)
                    )
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
                            + (
                                0.5 * eta * (k_0 + 1 + 2 * gamma_variable)
                                + gamma_variable * cos(phi)
                            )
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
            * (
                -gamma_variable * cos(x + phi)
                + x * sin(phi)
                + (
                    0.5 * eta * (k_0 + 1 + 2 * gamma_variable)
                    + gamma_variable * cos(phi)
                )
            )
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

    def eta_roll(x, eta, theta3):
        return (
            2
            * (
                -gamma * cos(x + phi)
                + x * sin(phi)
                + (
                    (
                        0.5 * eta * (k_0 + 1 + 2 * gamma * cos(theta3))
                        + gamma * cos(phi + theta3)
                        - theta3 * sin(phi)
                    )
                )
            )
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
                    * (
                        -gamma * cos(x + phi)
                        + x * sin(phi)
                        + (
                            (
                                0.5 * eta * (k_0 + 1 + 2 * gamma * cos(theta3))
                                + gamma * cos(phi + theta3)
                                - theta3 * sin(phi)
                            )
                        )
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
                    + (
                        (
                            0.5 * eta * (k_0 + 1 + 2 * gamma * cos(theta3))
                            + gamma * cos(phi + theta3)
                            - theta3 * sin(phi)
                        )
                    )
                )
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
                        * (
                            -gamma * cos(x + phi)
                            + x * sin(phi)
                            + (
                                (
                                    0.5 * eta * (k_0 + 1 + 2 * gamma * cos(theta3))
                                    + gamma * cos(phi + theta3)
                                    - theta3 * sin(phi)
                                )
                            )
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
                + (
                    (
                        0.5 * eta * (k_0 + 1 + 2 * gamma * cos(theta3))
                        + gamma * cos(phi + theta3)
                        - theta3 * sin(phi)
                    )
                )
            )
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
                        * (
                            -gamma * cos(x + phi)
                            + x * sin(phi)
                            + (
                                (
                                    0.5 * eta * (k_0 + 1 + 2 * gamma * cos(theta3))
                                    + gamma * cos(phi + theta3)
                                    - theta3 * sin(phi)
                                )
                            )
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
                        + (
                            (
                                0.5 * eta * (k_0 + 1 + 2 * gamma * cos(theta3))
                                + gamma * cos(phi + theta3)
                                - theta3 * sin(phi)
                            )
                        )
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
                                + (
                                    (
                                        0.5 * eta * (k_0 + 1 + 2 * gamma * cos(theta3))
                                        + gamma * cos(phi + theta3)
                                        - theta3 * sin(phi)
                                    )
                                )
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
                    + (
                        (
                            0.5 * eta * (k_0 + 1 + 2 * gamma * cos(theta3))
                            + gamma * cos(phi + theta3)
                            - theta3 * sin(phi)
                        )
                    )
                )
                / (2 * (1 + gamma * cos(x)))
                * cos(x)
            )
        )

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

    def F_spin(x, eta):
        return mu * N_spin(x, eta)

    def dXi_dt_spin(x, Xi, solution_y):
        func = solution_y.y[0, np.argmin(np.abs(solution_y.t - x))]
        return (
            sin(phi)
            - gamma * (zeta_spin(x, func) * cos(x) - func * sin(x))
            - mu * N_spin(x, func)
        ) / sqrt(np.abs((func)))

    # Skidding

    def zeta_skid(x, eta):
        return (
            (gamma * sin(x) + mu * (1 + gamma * cos(x)))
            * (cos(phi) - gamma * eta * cos(x))
            / (
                k_g
                + (gamma * sin(x) + mu * (1 + gamma * cos(x))) * gamma * sin(x)
                + 0.000000000001
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
                    + 0.000000000001
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
                        + 0.00000000001
                    )
                )
            )
            * sin(x)
            + eta * cos(x)
        )

    def F_skid(x, eta):
        return mu * N_skid(x, eta)

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
        ) / sqrt(np.abs((func)))

    def energy_lost(x, E, force, solution_eta, solution_velocity):
        eta = solution_eta.y[0, np.argmin(np.abs(solution_eta.t - x))]
        velocity = solution_velocity.y[0, np.argmin(np.abs(solution_velocity.t - x))]
        dEdx = force(x, eta) * velocity / (sqrt(np.abs((eta))))
        return dEdx

    eta_final = np.zeros(N)
    F_final = np.zeros(N)
    N_final = np.zeros(N)
    F_N_final = np.zeros(N)
    Xi_final = np.zeros(N)
    root_eta = np.zeros(N)
    friction_heat = 0
    theta_start = theta[0]
    roll = False
    spin = False
    skid = False
    sp = 0
    sk = 0
    indicator = 0
    indicator1 = 0

    eta_final[0] = eta_0
    F_final[0] = -0.5 * (1 - gamma) * sin(phi)
    N_final[0] = gamma * (crit_velocity - eta_0)
    F_N_final[0] = F_final[0] / N_final[0]
    Xi_final[0] = sqrt(eta_0)
    root_eta[0] = sqrt(eta_0)

    roll = True
    spin = False
    skid = False
    etaNew = eta_0
    for i in range(1, N):
        if N_final[i - 1] <= 0 and eta_final[i - 1] * cos(theta[i - 1]) > crit_velocity:
            indicator = 1
            break
        elif np.abs(F_N_final[i - 1]) >= mu:
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
                    thetaNew = theta[i:]
                    solskid = eta(
                        thetaNew[0],
                        dn_dx_skid,
                        theta[i - 1],
                        theta_f,
                        thetaNew,
                        etaNew,
                    )
                    Xi_initial = np.array([Xi_final[i - 1]])
                    solvel_skid = solve_ivp(
                        lambda x, Xi: dXi_dt_skid(x, Xi, solskid),
                        (theta[i - 1], theta_f),
                        Xi_initial,
                        t_eval=thetaNew,
                    )
                    solenergy_skid = solve_ivp(
                        lambda x, E: energy_lost(x, E, F_skid, solskid, solvel_skid),
                        (theta[i - 1], theta_f),
                        np.array([friction_heat]),
                        t_eval=thetaNew,
                    )
                    # print(friction_heat)
                    Xi_final[i] = solvel_skid.y[0, 0]
                    eta_final[i] = solskid.y[0, 0]
                    root_eta[i] = sqrt(eta_final[i])
                    N_final[i] = N_skid(thetaNew[0], eta_final[i])
                    F_final[i] = -mu * N_final[i]
                    F_N_final[i] = -mu
                elif skid == True:
                    roll = False
                    spin = False
                    sk = sk + 1
                    eta_final[i] = solskid.y[0, sk]
                    Xi_final[i] = solvel_skid.y[0, sk]
                    # print(eta_0 / 2)
                    # print("a")
                    # print(solenergy_skid.y[0, sk])
                    # print(solenergy_skid.y[0])
                    if solenergy_skid.y[0, sk] > eta_0:
                        indicator1 = 1
                    if eta_final[i] < 0:
                        # indicator1 = 1
                        break
                    root_eta[i] = sqrt(eta_final[i])
                    N_final[i] = N_skid(thetaNew[sk], eta_final[i])
                    F_final[i] = -mu * N_final[i]
                    F_N_final[i] = -mu
                    if Xi_final[i] < root_eta[i]:
                        skid = False
                        roll = False
                        spin = False
                        F_N_final[i] = -mu + 0.000000000000001
                        friction_heat = friction_heat + solenergy_skid.y[0, sk]
                        # print("b")
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
                    thetaNew = theta[i:]
                    solspin = eta(
                        thetaNew[0],
                        dn_dx_spin,
                        theta[i - 1],
                        theta_f,
                        thetaNew,
                        etaNew,
                    )
                    eta_final[i] = solspin.y[0, 0]
                    Xi_initial = np.array([Xi_final[i - 1]])
                    solvel_spin = solve_ivp(
                        lambda x, Xi: dXi_dt_spin(x, Xi, solspin),
                        (theta[i - 1], theta_f),
                        Xi_initial,
                        t_eval=thetaNew,
                    )
                    solenergy_spin = solve_ivp(
                        lambda x, E: energy_lost(x, E, F_spin, solspin, solvel_spin),
                        (theta[i - 1], theta_f),
                        np.array([friction_heat]),
                        t_eval=thetaNew,
                    )
                    # print(solenergy_spin.y[0])
                    # print(friction_heat)
                    Xi_final[i] = solvel_spin.y[0, 0]
                    root_eta[i] = sqrt(eta_final[i])
                    N_final[i] = N_spin(thetaNew[0], eta_final[i])
                    F_final[i] = mu * N_final[i]
                    F_N_final[i] = mu
                elif spin == True:
                    roll = False
                    skid = False
                    sp = sp + 1
                    eta_final[i] = solspin.y[0, sp]
                    Xi_final[i] = solvel_spin.y[0, sp]
                    # print(eta_0 / 2)
                    # print("b")
                    # print(solenergy_spin.y[0, sp])
                    if solenergy_spin.y[0, sp] > eta_0:
                        indicator1 = 1
                        # pass
                    if eta_final[i] < 0:
                        # indicator1 = 1
                        break
                    root_eta[i] = sqrt(eta_final[i])
                    N_final[i] = N_spin(thetaNew[sp], eta_final[i])
                    F_final[i] = mu * N_final[i]
                    F_N_final[i] = mu
                    if Xi_final[i] > root_eta[i]:
                        skid = False
                        roll = False
                        spin = False
                        F_N_final[i] = mu - 0.0000000000001
                        friction_heat = friction_heat + solenergy_spin.y[0, sp]
                        # print("a")
        else:
            if roll == False:
                roll = True
                skid = False
                spin = False
                etaNew = eta_final[i - 1]
                theta_start = theta[i]
                eta_final[i] = eta_roll(theta[i], etaNew, theta_start)
                root_eta[i] = sqrt(eta_final[i])
                F_final[i] = F_roll(theta[i], etaNew, theta_start)
                N_final[i] = N_roll(theta[i], etaNew, theta_start)
                F_N_final[i] = F_N_roll(theta[i], etaNew, theta_start)
                Xi_final[i] = sqrt(eta_final[i])
            elif roll == True:
                skid = False
                spin = False
                eta_final[i] = eta_roll(theta[i], etaNew, theta_start)
                if eta_final[i] < 0:
                    indicator1 = 1
                    break
                root_eta[i] = sqrt(eta_final[i])
                F_final[i] = F_roll(theta[i], etaNew, theta_start)
                N_final[i] = N_roll(theta[i], etaNew, theta_start)
                F_N_final[i] = F_N_roll(theta[i], etaNew, theta_start)
                Xi_final[i] = sqrt(eta_final[i])

    return indicator, indicator1


def calculate_slipping_boundary(x_pts, vel_resolution, i):
    nu = x_pts[i]
    gamma = 1 / (1 + nu)
    upper = crit_velocity(nu)
    lower = 0
    for j in range(vel_resolution):
        best_guess = avg(upper, lower)
        if (
            np.abs(np.max(F_N_Phase(thetaPhase, best_guess, gamma))) >= mu
            or np.abs(np.min(F_N_Phase(thetaPhase, best_guess, gamma))) >= mu
        ):
            upper = best_guess
        else:
            lower = best_guess
    return nu, avg(upper, lower)


def calculate_hopping_boundary(x_pts, vel_resolution, angular_resolution, i):
    nu = x_pts[i]
    upper = crit_velocity(nu)
    lower = 0
    upper1 = crit_velocity(nu)
    lower1 = 0
    for j in range(vel_resolution):
        best_guess = avg(upper, lower)
        best_guess1 = avg(upper1, lower1)
        a, b = x_parallel(phi, nu, best_guess, mu, angular_resolution)
        if a == 1:
            upper = best_guess
        else:
            lower = best_guess
        if b == 1:
            lower1 = best_guess1
        elif b == 0:
            upper1 = best_guess1
    return nu, avg(upper, lower), avg(upper1, lower1)


if __name__ == "__main__":

    n = 150  # Mass resolution
    m = 15  # Velocity resolution 2^m
    N = 3600  # Angular resolution
    mass_ratio = linspace(0.0000000006, 0.5, n)

    # Using Pool for managing processes
    with Pool() as pool:
        hopping_results = pool.starmap(
            calculate_hopping_boundary,
            [(mass_ratio, m, N, i) for i in range(n)],
        )
        slipping_results = pool.starmap(
            calculate_slipping_boundary,
            [(mass_ratio, m, i) for i in range(n)],
        )

plot_pts = list(zip(*hopping_results))

# Record the end time
end_time = time.time()


# Calculate the runtime
runtime = end_time - start_time

print(f"Runtime: {runtime} seconds")

# plt.plot(*zip(*slipping_results))
plt.plot([0, 1], [crit_velocity(0), crit_velocity(1)])
plt.plot(plot_pts[0], plot_pts[1])
plt.plot(plot_pts[0], plot_pts[2])
plt.ylim(0, 2)
plt.xlim(0, 1)
current_time = datetime.now()
plotname = (
    str(n)
    + " "
    + str(m)
    + " "
    + str(N)
    + " "
    + str(mu)
    + " "
    + str(phi)
    + str(current_time)
    + ".png"
)
plt.savefig(plotname, format="png")
plt.show()

# Open a file for writing
with open("III.0.6.0.txt", "a") as file:
    file.write("X Divisions: " + str(n) + "\n")
    file.write("Y Divisions: " + str(m) + "\n")
    file.write("Angle Divisions: " + str(N) + "\n")
    file.write(str(hopping_results) + "\n")
    file.write("\n")
file.close()
