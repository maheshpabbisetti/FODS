import numpy as np
from sklearn.linear_model import LogisticRegression

def get_user_input():
   
    print("Enter the features of the new customer:")
    usage_minutes = float(input("Usage Minutes: "))
    contract_duration = float(input("Contract Duration (months): "))

    return [[usage_minutes, contract_duration]] 
def main():
    dataset = np.array([
        [1000, 12, 0],
        [500, 6, 1],
        [1500, 24, 0],
    ])

    X = dataset[:, :-1]
    y = dataset[:, -1]

    new_customer_features = get_user_input()

    log_reg = LogisticRegression()
    log_reg.fit(X, y)

    prediction = log_reg.predict(new_customer_features)

    if prediction[0] == 1:
        print("The new customer is likely to churn.")
    else:
        print("The new customer is unlikely to churn.")

if __name__ == "__main__":
    main()
