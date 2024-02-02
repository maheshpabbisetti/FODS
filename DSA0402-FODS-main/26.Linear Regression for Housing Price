import numpy as np
from sklearn.linear_model import LinearRegression

def get_user_input():
    print("Enter the features of the new house:")
    area = float(input("Area (in square feet): "))
    num_bedrooms = int(input("Number of Bedrooms: "))

    return [[area, num_bedrooms]]  

def main():

    dataset = np.array([
        [1500, 3, 200000],
        [1800, 4, 250000],
        [1200, 2, 150000],
       
    ])
    X = dataset[:, :-1]
    y = dataset[:, -1]

    new_house_features = get_user_input()

    linear_reg = LinearRegression()
    linear_reg.fit(X, y)

    prediction = linear_reg.predict(new_house_features)

    print(f"The predicted price of the new house is: ${prediction[0]:.2f}")

if __name__ == "__main__":
    main()
