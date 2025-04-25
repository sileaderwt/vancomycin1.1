import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Define the Two-compartment IV bolus model
def two_compartment_model(t, params, weight, amount):
    """
    Calculate the concentration of drug in two-compartments after an IV bolus dose
    t: time array (hours)
    params: model parameters [V1, V2, CL1, CL2, Q]
        - V1: Volume of distribution of compartment 1 (L)
        - V2: Volume of distribution of compartment 2 (L)
        - CL1: Clearance from compartment 1 (L/h)
        - CL2: Clearance from compartment 2 (L/h)
        - Q: Inter-compartmental clearance (L/h)
    weight: Patient's weight (kg)
    amount: Dose amount (mg)
    """
    # Scale the parameters by weight (assuming 70 kg as the reference weight)
    V1, V2, CL1, CL2, Q = params
    V1 *= weight / 70
    V2 *= weight / 70
    CL1 *= weight / 70
    CL2 *= weight / 70

    # Define the elimination rate constants
    k10 = CL1 / V1  # Elimination rate constant for compartment 1
    k12 = Q / V1  # Rate constant for transfer from compartment 1 to 2
    k21 = Q / V2  # Rate constant for transfer from compartment 2 to 1

    # Solve the system of differential equations (approximated numerically here)
    # Concentrations in the two compartments at time t
    C1 = (np.exp(-k10 * t) * (V1 / (V1 - V2)) + np.exp(-k12 * t) * (V2 / (V2 - V1)))  # Compartment 1 concentration
    C2 = (np.exp(-k21 * t) * (V2 / (V2 - V1)) + np.exp(-k10 * t) * (V1 / (V1 - V2)))  # Compartment 2 concentration

    # Total concentration from both compartments
    total_conc = C1 + C2

    # Adjust the initial dose (amount) over time (assuming instant IV bolus)
    total_conc = amount / (V1 + V2) * total_conc  # Adjust concentration based on dose

    return total_conc


# Define population parameter estimation
def population_parameter_estimation(data, initial_params):
    """
    Estimate population parameters by minimizing the objective function (e.g., sum of squared errors)
    data: Observed data (time, concentration, patient)
    initial_params: Initial guess for population parameters
    """

    # Define the objective function to minimize (sum of squared errors)
    def objective(params):
        error = 0
        for i, patient_data in data.groupby('patient'):
            t = patient_data['time'].values
            observed_conc = patient_data['conc'].values
            weight = patient_data['weight'].iloc[0]  # Assuming constant weight for each patient
            amount = patient_data['amt'].iloc[0]  # Assuming constant dose for each patient
            predicted_conc = two_compartment_model(t, params, weight, amount)
            error += np.sum((observed_conc - predicted_conc) ** 2)
        return error

    # Optimize the parameters to minimize the error
    result = minimize(objective, initial_params, method='L-BFGS-B')
    return result.x  # Return the estimated population parameters


# Individual parameter estimation
def individual_parameter_estimation(patient_data, initial_params):
    """
    Estimate individual parameters for a single patient
    patient_data: Data for a single patient (time, concentration)
    initial_params: Initial guess for individual parameters
    """
    t = patient_data['time'].values
    observed_conc = patient_data['conc'].values
    weight = patient_data['weight'].iloc[0]  # Assuming constant weight for each patient
    amount = patient_data['amt'].iloc[0]  # Assuming constant dose for each patient

    # Define the objective function for individual parameter estimation
    def objective(params):
        predicted_conc = two_compartment_model(t, params, weight, amount)
        return np.sum((observed_conc - predicted_conc) ** 2)

    # Optimize the individual parameters for this patient
    result = minimize(objective, initial_params, method='L-BFGS-B')
    return result.x


# Shrinkage calculation
def calculate_shrinkage(individual_params, population_params):
    """
    Calculate shrinkage based on the difference between individual and population parameters
    individual_params: Array of individual parameter estimates
    population_params: Array of population parameter estimates
    """
    shrinkage = 100 * np.mean(np.abs(individual_params - population_params) / population_params)
    return shrinkage


# Example usage
# Simulated data (replace with real patient data)
# Assume patient data with columns: 'patient', 'time', 'conc', 'weight', 'amt'
data = pd.DataFrame({
    'patient': [1, 1, 1, 2, 2, 2],
    'time': [0, 1, 2, 0, 1, 2],
    'conc': [10, 5, 2, 12, 6, 3],
    'weight': [70, 70, 70, 80, 80, 80],  # Weight of each patient in kg
    'amt': [1000, 1000, 1000, 1200, 1200, 1200]  # Dose amount (mg)
})

# Initial guess for population parameters [V1, V2, CL1, CL2, Q]
initial_population_params = [10, 20, 5, 3, 2]

# Estimate population parameters
population_params = population_parameter_estimation(data, initial_population_params)
print("Population Parameter Estimates:", population_params)

# Estimate individual parameters for each patient
individual_params = []
for patient_id in data['patient'].unique():
    patient_data = data[data['patient'] == patient_id]
    individual_params.append(individual_parameter_estimation(patient_data, initial_population_params))

# Calculate shrinkage
shrinkage = calculate_shrinkage(np.array(individual_params), population_params)
print(f"Shrinkage: {shrinkage}%")
