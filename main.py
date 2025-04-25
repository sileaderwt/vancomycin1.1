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


def plot_observed_vs_individual(data, individual_params):
    plt.figure(figsize=(10, 6))

    all_preds = []
    all_obs = []

    for pid, pdata in data.groupby('patient'):
        # Get individual predictions
        times, preds = solve_two_compartment(pdata, individual_params[pid - 1])

        # Observed concentrations (evid == 0)
        obs_data = pdata[pdata['evid'] == 0]
        obs_concs = obs_data['conc'].values

        # Store for setting axis limits
        all_preds.extend(preds)
        all_obs.extend(obs_concs)

        # Plot observed vs individual predicted
        plt.scatter(preds, obs_concs, label=f'Patient {pid}', alpha=0.6)

    # Diagonal identity line
    min_val = min(min(all_preds), min(all_obs))
    max_val = max(max(all_preds), max(all_obs))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Identity Line')

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

    all_preds = []
    all_obs = []

    for pid, pdata in data.groupby('patient'):
        # Get population model predictions
        times, preds = solve_two_compartment(pdata, pop_params)

        # Observed concentrations (evid == 0)
        obs_data = pdata[pdata['evid'] == 0]
        obs_concs = obs_data['conc'].values

        # Store all points for setting axis limits
        all_preds.extend(preds)
        all_obs.extend(obs_concs)

        # Plot observed vs population predicted
        plt.scatter(preds, obs_concs, label=f'Patient {pid}', alpha=0.6)

    # Diagonal identity line
    min_val = min(min(all_preds), min(all_obs))
    max_val = max(max(all_preds), max(all_obs))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Identity Line')

    plt.xlabel("Population Predicted Concentration (ng/mL)")
    plt.ylabel("Observed Concentration (ng/mL)")
    plt.title("Observed vs Population Predicted Concentrations")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

# Plot conditional weighted residuals
def plot_cwres(data, individual_params, population_params):
    # Plot Individual CWRES
    plt.figure(figsize=(10, 6))
    for pid, pdata in data.groupby('patient'):
        times, individual_preds = solve_two_compartment(pdata, individual_params[pid - 1])
        obs_data = pdata[pdata['evid'] == 0]
        obs_concs = obs_data['conc'].values
        individual_residuals = obs_concs - individual_preds
        individual_cwres = individual_residuals  # assuming sigma = 1

        plt.plot(times, individual_cwres, 'o', label=f'Patient {pid}', alpha=0.6)

    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel("Time Post Dose (h)")
    plt.ylabel("Conditional Weighted Residuals (Individual)")
    plt.title("CWRES for Individual Predictions")
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()

    # Plot Population CWRES
    plt.figure(figsize=(10, 6))
    for pid, pdata in data.groupby('patient'):
        _, population_preds = solve_two_compartment(pdata, population_params)
        obs_data = pdata[pdata['evid'] == 0]
        obs_concs = obs_data['conc'].values
        population_residuals = obs_concs - population_preds
        population_cwres = population_residuals  # assuming sigma = 1

        times = pdata[pdata['evid'] == 0]['time'].values
        plt.plot(times, population_cwres, 'o', label=f'Patient {pid}', alpha=0.6)

    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel("Time Post Dose (h)")
    plt.ylabel("Conditional Weighted Residuals (Population)")
    plt.title("CWRES for Population Predictions")
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()


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



###BOX PLOT
# Define the therapeutic range (e.g., 5–10 mg/L)
THERAPEUTIC_RANGE_MIN = 5.0
THERAPEUTIC_RANGE_MAX = 10.0

# Function to determine the proportion in the therapeutic range
def proportion_in_therapeutic_range(concentrations):
    return np.sum((concentrations >= THERAPEUTIC_RANGE_MIN) & (concentrations <= THERAPEUTIC_RANGE_MAX)) / len(concentrations)

# Simulate dose adjustments and calculate proportions in the therapeutic range
def simulate_proportions_in_therapeutic_range(data, pop_params):
    proportions = []
    for pid, pdata in data.groupby('patient'):
        times, concs = solve_two_compartment(pdata, pop_params)
        proportion = proportion_in_therapeutic_range(concs)
        proportions.append(proportion)
    return proportions

# Generate the boxplot of the proportions in the therapeutic range
def plot_boxplot_of_proportions(proportions):
    print(proportions)
    plt.figure(figsize=(8, 6))
    plt.boxplot(proportions, vert=False)
    plt.title('Proportion of Patients in Therapeutic Range')
    plt.xlabel('Proportion in Therapeutic Range')
    plt.grid(True)
    plt.show()
###BOXPLOT END

# Example data
data = pd.read_excel("sample1.xlsx")

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
plot_cwres(data, individuals, pop_params)
# Perform Visual Predictive Check
vpc(data, pop_params, n_simulations=1000, percentile_range=(5, 50, 95))

#box plot
proportions = simulate_proportions_in_therapeutic_range(data, pop_params)
plot_boxplot_of_proportions(proportions)