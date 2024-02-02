import numpy as np
from scipy import stats

drug_data = np.array([20,39,48,58,48,38,58,49,39,69,59,68,79,96,83]) 
placebo_data = np.array([90,80,90,89,79,69,80,70,80,80,80,70,90,70,80]) 

drug_mean = np.mean(drug_data)
drug_std = np.std(drug_data, ddof=1)

placebo_mean = np.mean(placebo_data)
placebo_std = np.std(placebo_data, ddof=1)

n_drug = len(drug_data)
n_placebo = len(placebo_data)

df = n_drug - 1

t_score = stats.t.ppf(0.975, df)

se_drug = drug_std / np.sqrt(n_drug)
se_placebo = placebo_std / np.sqrt(n_placebo)

margin_of_error_drug = t_score * se_drug
margin_of_error_placebo = t_score * se_placebo

confidence_interval_drug = (drug_mean - margin_of_error_drug, drug_mean + margin_of_error_drug)
confidence_interval_placebo = (placebo_mean - margin_of_error_placebo, placebo_mean + margin_of_error_placebo)

print("95% Confidence Interval for Mean Reduction in Blood Pressure (Drug Group):", confidence_interval_drug)
print("95% Confidence Interval for Mean Reduction in Blood Pressure (Placebo Group):", confidence_interval_placebo)
