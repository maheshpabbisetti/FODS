import scipy.stats as stats

# Replace these with your actual data
conversion_rates_design_A = [0.05, 0.1, 0.08, 0.12, 0.09, ...]  # Sample data for design A
conversion_rates_design_B = [0.07, 0.11, 0.09, 0.13, 0.1, ...]  # Sample data for design B

# Step 2: Calculate Descriptive Statistics
mean_A = np.mean(conversion_rates_design_A)
std_dev_A = np.std(conversion_rates_design_A, ddof=1)
n_A = len(conversion_rates_design_A)

mean_B = np.mean(conversion_rates_design_B)
std_dev_B = np.std(conversion_rates_design_B, ddof=1)
n_B = len(conversion_rates_design_B)

# Step 3: Check Assumptions (Assume normality and homogeneity of variances for simplicity)

# Step 4: Set Significance Level
alpha = 0.05

# Step 5: Conduct T-Test
t_statistic, p_value = stats.ttest_ind(conversion_rates_design_A, conversion_rates_design_B, equal_var=False)

# Step 6: Determine Critical Region
# You can also use t-distribution tables or scipy.stats.t.ppf to find the critical value

# For a two-tailed test:
critical_value = stats.t.ppf(1 - alpha / 2, df=min(n_A - 1, n_B - 1))

# For a one-tailed test (left-tailed or right-tailed):
# critical_value = stats.t.ppf(1 - alpha, df=min(n_A - 1, n_B - 1))

# Step 7: Make a Decision
if abs(t_statistic) > critical_value or p_value < alpha:
    print("Reject the null hypothesis. There is a statistically significant difference.")
else:
    print("Fail to reject the null hypothesis. There is no statistically significant difference.")

# Step 8: Conclusion
print(f"T-Statistic: {t_statistic}")
print(f"P-Value: {p_value}")
