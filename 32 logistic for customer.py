import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Assuming you have a customer dataset with features and churn status (replace this with your dataset)
# Here, I'll create a sample dataset for demonstration purposes
customer_data = np.array([
    [100, 12, 1, 0],   # Customer 1: usage minutes, contract duration, is_employee, churn status
    [200, 6, 0, 0],    # Customer 2
    [50, 24, 1, 1],    # Customer 3
    [150, 18, 0, 0],   # Customer 4
    [120, 9, 1, 1]     # Customer 5
])

# Split the data into features and labels
X = customer_data[:, :-1]  # Features
y = customer_data[:, -1]   # Churn status

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
logistic_regression_model = LogisticRegression(random_state=42)
logistic_regression_model.fit(X_train, y_train)

def predict_churn(new_customer_features, model, scaler):
    # Standardize the new customer features
    new_customer_features = scaler.transform([new_customer_features])

    # Predict churn status (0 for not churned, 1 for churned)
    prediction = model.predict(new_customer_features)
    return prediction[0]

if __name__ == "__main__":
    # Get user input for new customer features
    new_customer_features = [float(feature) for feature in input("Enter new customer features separated by commas: ").split(',')]

    # Predict whether the new customer will churn or not
    churn_prediction = predict_churn(new_customer_features, logistic_regression_model, scaler)

    # Display the result
    if churn_prediction == 0:
        print("The new customer is predicted not to churn.")
    else:
        print("The new customer is predicted to churn.")
