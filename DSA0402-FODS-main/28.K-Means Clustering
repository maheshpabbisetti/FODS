import numpy as np
from sklearn.cluster import KMeans

def get_user_input(num_features):
    print("Enter the shopping-related features of the new customer:")
    new_customer_features = []
    for i in range(num_features):
        feature = float(input(f"Feature {i+1}: "))
        new_customer_features.append(feature)

    return np.array([new_customer_features])

def main():
    dataset = np.array([
        [10, 20],
        [5, 15],
        [30, 40],
    ])

    X = dataset  
    num_clusters = 3

    num_features = X.shape[1]
    new_customer_features = get_user_input(num_features)

    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)
    cluster_labels = kmeans.predict(X)

    new_customer_cluster = kmeans.predict(new_customer_features)

    print("The new customer is assigned to cluster:", new_customer_cluster[0])

if __name__ == "__main__":
    main()
