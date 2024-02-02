import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Assuming you have a housing dataset with features (area, bedrooms, etc.) and prices (replace this with your dataset)
# Here, I'll create a sample dataset for demonstration purposes
housing_data = np.array([
    [1500, 3, 2, 300000],   # House 1: area, bedrooms, bathrooms, price
    [2000, 4, 3, 400000],   # House 2
    [1200, 2, 1, 250000],   # House 3
    [1800, 3, 2, 350000],   # House 4
    [1600, 3, 2, 320000]    # House 5
])

# Split the data into features and prices
X = housing_data[:, :-1]  # Features
y = housing_data[:, -1]   # Prices

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)

def predict_house_price(new_house_features, model, scaler):
    # Standardize the new house features
    new_house_features = scaler.transform([new_house_features])

    # Predict the house price
    predicted_price = model.predict(new_house_features)
    return predicted_price[0]

if __name__ == "__main__":
    # Get user input for new house features
    new_house_features = [float(feature) for feature in input("Enter new house features separated by commas: ").split(',')]

    # Predict the price of the new house
    predicted_price = predict_house_price(new_house_features, linear_regression_model, scaler)

    # Display the result
    print(f"The predicted price of the new house is: ${predicted_price:.2f}")
