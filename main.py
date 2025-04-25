import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


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


# Shrinkage
def calculate_shrinkage(individual_params, population_params):
    return 100 * np.mean(np.abs(np.array(individual_params) - np.array(population_params)) / population_params)


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


# Visual Predictive Check (VPC)
def vpc(data, pop_params, n_simulations=1000, percentile_range=(5, 50, 95)):
    all_simulated_concs = []

    # Simulate data for multiple virtual subjects
    for _ in range(n_simulations):
        simulated_concs = []
        for pid, pdata in data.groupby('patient'):
            times, preds = solve_two_compartment(pdata, pop_params)
            simulated_concs.append(preds)

        all_simulated_concs.append(np.concatenate(simulated_concs))

    # Calculate percentiles of the simulated concentrations
    all_simulated_concs = np.concatenate(all_simulated_concs)
    lower_percentile = np.percentile(all_simulated_concs, percentile_range[0])
    median_percentile = np.percentile(all_simulated_concs, percentile_range[1])
    upper_percentile = np.percentile(all_simulated_concs, percentile_range[2])

    # Plot observed vs simulated concentrations (VPC)
    plt.figure(figsize=(8, 6))

    # Observed data (for reference)
    obs_data = data[data['evid'] == 0]
    plt.scatter(obs_data['time'], obs_data['conc'], color='black', alpha=0.6, label='Observed')

    # Simulated data percentiles
    plt.plot(obs_data['time'], np.repeat(median_percentile, len(obs_data)), 'b--', label='50th Percentile (Median)')
    plt.fill_between(obs_data['time'], lower_percentile, upper_percentile, color='gray', alpha=0.5,
                     label='5th-95th Percentile')

    plt.xlabel("Time (h)")
    plt.ylabel("Concentration (ng/mL)")
    plt.title("Visual Predictive Check (VPC)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Example data (same as previous code)
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

# Perform Visual Predictive Check
vpc(data, pop_params, n_simulations=1000, percentile_range=(5, 50, 95))

