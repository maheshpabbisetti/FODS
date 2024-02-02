import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree classifier
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

def predict_flower_species(new_flower_features, model):
    # Predict the species of the new flower
    predicted_species = model.predict([new_flower_features])
    return predicted_species[0]

if __name__ == "__main__":
    # Get user input for new flower features
    new_flower_features = [float(feature) for feature in input("Enter new flower features separated by commas: ").split(',')]

    # Predict the species of the new flower
    predicted_species = predict_flower_species(new_flower_features, decision_tree_model)

    # Display the result
    species_names = iris.target_names
    print(f"The predicted species of the new flower is: {species_names[predicted_species]}")
