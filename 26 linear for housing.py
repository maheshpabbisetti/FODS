import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load and Explore Data
# Load your house dataset (replace 'your_house_data.csv' with your actual data file)
house_data = pd.read_csv('your_house_data.csv')

# Display the first few rows of the DataFrame
print(house_data.head())

# Step 2: Bivariate Analysis
# Select the feature (in this case, house size) and the target variable (house price)
feature = 'house_size'
target = 'house_price'

# Plot a scatter plot for bivariate analysis
plt.scatter(house_data[feature], house_data[target], alpha=0.5)
plt.title(f'Bivariate Analysis: {feature} vs {target}')
plt.xlabel(feature)
plt.ylabel(target)
plt.show()

# Step 3: Prepare Data for Linear Regression
X = house_data[[feature]]
y = house_data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate Model Performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display model performance metrics
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2): {r2:.2f}')

# Step 7: Visualize the Regression Line
plt.scatter(X_test, y_test, alpha=0.5, label='Actual Prices')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.title('Linear Regression Model: Actual Prices vs Predicted Prices')
plt.xlabel(feature)
plt.ylabel(target)
plt.legend()
plt.show()
