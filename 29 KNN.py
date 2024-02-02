import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Assuming you have a medical dataset with features and labels (replace this with your dataset)
# Here, I'll create a sample dataset for demonstration purposes
medical_data = np.array([
    [70, 160, 1],   # Patient 1: age, weight, condition label
    [45, 120, 0],   # Patient 2
    [60, 150, 1],   # Patient 3
    [55, 130, 0],   # Patient 4
    [80, 180, 1]    # Patient 5
])

# Split the data into features and labels
X = medical_data[:, :-1]  # Features
y = medical_data[:, -1]   # Condition labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get user input for new patient features and k value
new_patient_features = [float(feature) for feature in input("Enter new patient features separated by commas (e.g., age, weight): ").split(',')]
k_value = int(input("Enter the value of k for KNN: "))

# Train the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=k_value)
knn_classifier.fit(X_train, y_train)

def predict_medical_condition(new_patient_features, model):
    # Predict the medical condition of the new patient
    predicted_condition = model.predict([new_patient_features])
    return predicted_condition[0]

if __name__ == "__main__":
    # Predict the medical condition of the new patient
    predicted_condition = predict_medical_condition(new_patient_features, knn_classifier)

    # Display the result
    if predicted_condition == 0:
        print("The new patient is predicted not to have the medical condition.")
    else:
        print("The new patient is predicted to have the medical condition.")
