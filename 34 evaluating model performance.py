import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris  # Replace with your dataset
from sklearn.ensemble import RandomForestClassifier  # Replace with your model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data():
    # Load your dataset here; replace this with your actual dataset loading code
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

def train_model(X_train, y_train):
    # Train your model here; replace this with your actual model training code
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Display the results
    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

if __name__ == "__main__":
    # Load data
    X, y = load_data()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Get user input for feature names and target variable
    features = input("Enter feature names separated by commas: ").split(',')
    target = input("Enter the target variable name: ")

    # Index features and target in the dataset
    feature_indices = [iris.feature_names.index(feature.strip()) for feature in features]
    target_index = iris.feature_names.index(target.strip())

    # Evaluate the model
    evaluate_model(model, X_test[:, feature_indices], y_test)
