import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Two-compartment ODE system
def two_compartment_ode(t, y, V1, V2, CL1, CL2, Q):
    A1, A2 = y
    dA1dt = - (CL1 / V1) * A1 - Q * (A1 / V1 - A2 / V2)
    dA2dt = Q * (A1 / V1 - A2 / V2)
    return [dA1dt, dA2dt]


# Solve the system for one patient
def solve_two_compartment(patient_data, base_params):
    times = patient_data['time'].values
    amts = patient_data['amt'].values
    evids = patient_data['evid'].values

    weight = patient_data['weight'].iloc[0]
    age = patient_data['age'].iloc[0]
    scr = patient_data['scr'].iloc[0]
    gender = patient_data['gender'].iloc[0]  # 0 for male, 1 for female
    sex_factor = 0.85 if gender == 1 else 1.0

    # Covariate-adjusted parameters
    V1, V2, CL1, CL2, Q = base_params
    V1 *= weight / 70
    V2 *= weight / 70
    CL2 *= weight / 70
    Q *= weight / 70

    # Estimate CLcr using Cockcroft-Gault
    clcr = ((140 - age) * weight) / (72 * scr) * sex_factor  # mL/min
    clcr_Lph = clcr * 60 / 1000  # Convert to L/h

    CL1 = clcr_Lph  # Assume CL1 is renal clearance

    A1, A2 = 0, 0
    result_times = []
    result_concs = []

    for i in range(1, len(times)):
        t_start = times[i - 1]
        t_end = times[i]

        if evids[i - 1] == 1:
            A1 += amts[i - 1]

        sol = solve_ivp(
            fun=two_compartment_ode,
            t_span=[t_start, t_end],
            y0=[A1, A2],
            args=(V1, V2, CL1, CL2, Q),
            t_eval=[t_end]
        )

        A1, A2 = sol.y[:, -1]
        result_times.append(t_end)
        result_concs.append(A1 / V1)

    return np.array(result_times), np.array(result_concs)


# Population objective function
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


# Shrinkage
def calculate_shrinkage(individual_params, population_params):
    return 100 * np.mean(np.abs(np.array(individual_params) - np.array(population_params)) / population_params)


# Define multiple new patients
def predict_batch_new_patients(new_data, pop_params):
    all_preds = []
    for pid, pdata in new_data.groupby('patient'):
        times, preds = solve_two_compartment(pdata, pop_params)
        pred_df = pd.DataFrame({
            'patient': pid,
            'time': times,
            'pred_conc': preds
        })
        all_preds.append(pred_df)
    return pd.concat(all_preds, ignore_index=True)


# Plot observed vs individual prediction
def plot_observed_vs_individual(data, individual_params):
    plt.figure(figsize=(10, 6))

    for pid, pdata in data.groupby('patient'):
        # Get individual predictions
        times, preds = solve_two_compartment(pdata, individual_params[pid - 1])

        # Observed concentrations (evid == 0)
        obs_data = pdata[pdata['evid'] == 0]

        # Plot observed vs individual predicted
        plt.scatter(preds, obs_data['conc'], label=f'Patient {pid}', alpha=0.6)

    plt.xlabel("Individual Predicted Concentration (ng/mL)")
    plt.ylabel("Observed Concentration (ng/mL)")
    plt.title("Observed vs Individual Predicted Concentrations")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()


# Plot observed vs population prediction
def plot_observed_vs_population(data, pop_params):
    plt.figure(figsize=(10, 6))

    for pid, pdata in data.groupby('patient'):
        # Get population model predictions
        times, preds = solve_two_compartment(pdata, pop_params)

        # Observed concentrations (evid == 0)
        obs_data = pdata[pdata['evid'] == 0]

        # Plot observed vs population predicted
        plt.scatter(preds, obs_data['conc'], label=f'Patient {pid}', alpha=0.6)

    plt.xlabel("Population Predicted Concentration (ng/mL)")
    plt.ylabel("Observed Concentration (ng/mL)")
    plt.title("Observed vs Population Predicted Concentrations")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()


# Example data
data = pd.DataFrame({
    'patient': [1, 1, 1, 2, 2, 2],
    'time': [0, 1, 2, 0, 1, 2],
    'amt': [1000, 0, 0, 1200, 0, 0],
    'evid': [1, 0, 0, 1, 0, 0],
    'conc': [np.nan, 6.0, 2.8, np.nan, 7.1, 3.5],
    'weight': [70, 70, 70, 80, 80, 80],
    'scr': [1.2, 1.2, 1.2, 1.0, 1.0, 1.0],
    'age': [45, 45, 45, 60, 60, 60],
    'gender': [0, 0, 0, 1, 1, 1]  # 0 = male, 1 = female
})

# Initial guess: V1, V2, CL1, CL2, Q
initial_params = [10, 20, 5, 3, 2]

# Run population parameter estimation
pop_params = estimate_population_parameters(data, initial_params)
print("Estimated Population Parameters:", pop_params)

# Estimate individual parameters
individuals = []
for pid in data['patient'].unique():
    pdata = data[data['patient'] == pid]
    ind_params = estimate_individual_parameters(pdata, pop_params)
    individuals.append(ind_params)

# Calculate shrinkage
shrinkage = calculate_shrinkage(individuals, pop_params)
print(f"Shrinkage: {shrinkage:.2f}%")

# Plot the observed vs individual predicted concentrations
plot_observed_vs_individual(data, individuals)

# Plot the observed vs population predicted concentrations
plot_observed_vs_population(data, pop_params)
