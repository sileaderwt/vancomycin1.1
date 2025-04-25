import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Two-compartment IV bolus model ODE system
def two_compartment_ode(t, y, V1, V2, CL1, CL2, Q):
    A1, A2 = y  # A1 = amount in central, A2 = amount in peripheral
    dA1dt = - (CL1 / V1) * A1 - Q * (A1 / V1 - A2 / V2)
    dA2dt = Q * (A1 / V1 - A2 / V2)
    return [dA1dt, dA2dt]


# Solve model over a series of events (dose/observation)
def solve_two_compartment(patient_data, params):
    V1, V2, CL1, CL2, Q = params
    times = patient_data['time'].values
    amts = patient_data['amt'].values
    evids = patient_data['evid'].values

    A1, A2 = 0, 0  # initial amounts
    result_times = []
    result_concs = []

    for i in range(1, len(times)):
        t_start = times[i - 1]
        t_end = times[i]
        dt = t_end - t_start

        if evids[i - 1] == 1:
            A1 += amts[i - 1]  # Administer dose into central compartment

        sol = solve_ivp(
            fun=two_compartment_ode,
            t_span=[t_start, t_end],
            y0=[A1, A2],
            args=(V1, V2, CL1, CL2, Q),
            t_eval=[t_end]
        )

        A1, A2 = sol.y[:, -1]  # update amounts
        result_times.append(t_end)
        result_concs.append(A1 / V1)  # only report central compartment concentration

    return np.array(result_times), np.array(result_concs)


# Objective function for population parameter estimation
def population_objective(params, data):
    total_error = 0
    for patient_id, patient_data in data.groupby('patient'):
        pred_times, pred_concs = solve_two_compartment(patient_data, params)
        obs_data = patient_data[patient_data['evid'] == 0]
        obs_concs = obs_data['conc'].values
        error = np.sum((obs_concs - pred_concs) ** 2)
        total_error += error
    return total_error


# Estimate population parameters
def estimate_population_parameters(data, initial_params):
    result = minimize(population_objective, initial_params, args=(data,), method='L-BFGS-B')
    return result.x


# Estimate individual parameters
def estimate_individual_parameters(patient_data, initial_params):
    def individual_objective(params):
        _, pred_concs = solve_two_compartment(patient_data, params)
        obs_concs = patient_data[patient_data['evid'] == 0]['conc'].values
        return np.sum((obs_concs - pred_concs) ** 2)

    result = minimize(individual_objective, initial_params, method='L-BFGS-B')
    return result.x


# Calculate shrinkage
def calculate_shrinkage(individual_params, population_params):
    return 100 * np.mean(np.abs(np.array(individual_params) - np.array(population_params)) / population_params)


# Example dataset with EVID, dose, and observations
data = pd.DataFrame({
    'patient': [1, 1, 1, 2, 2, 2],
    'time': [0, 1, 2, 0, 1, 2],
    'amt': [1000, 0, 0, 1200, 0, 0],  # Only dose at time 0
    'evid': [1, 0, 0, 1, 0, 0],  # Dose (1) at t=0, then observations (0)
    'conc': [np.nan, 5.0, 2.5, np.nan, 6.0, 3.0]
})

# Initial guess: V1, V2, CL1, CL2, Q
initial_params = [10, 20, 5, 3, 2]

# Estimate population parameters
pop_params = estimate_population_parameters(data, initial_params)
print("Estimated Population Parameters:", pop_params)

# Estimate individual parameters and compute shrinkage
individuals = []
for pid in data['patient'].unique():
    pdata = data[data['patient'] == pid]
    ind_params = estimate_individual_parameters(pdata, pop_params)
    individuals.append(ind_params)

shrinkage = calculate_shrinkage(individuals, pop_params)
print(f"Shrinkage: {shrinkage:.2f}%")
