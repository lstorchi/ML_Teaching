import numpy as np
import pandas as pd
from pysr import PySRRegressor
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    df = pd.read_csv('data/surface.csv')
    df = df.drop('dE', axis=1)  
    print(df.shape)
    print(df.head())

    v = df['v\T']
    T = df.columns[1:]
    T = [int(t) for t in T]
    v = [int(v) for v in v]
    print(v)
    print(T)

    x, y = np.meshgrid(T, v)
    print(x.shape)
    print(y.shape)
    z = np.array(df.iloc[:,1:])
    z = np.log10(z)
    print(z.shape)

    fig = plt.figure()
    fig.set_size_inches(12, 12)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color='b')
    ax.set_xlabel('T (K)')
    ax.set_ylabel('v (quantum number)')
    ax.set_zlabel('k (cm^3/s)')
    plt.savefig('surface.png')

    t = x.reshape(-1)
    v = y.reshape(-1)
    X = np.column_stack((t, v)) 
    Y = z.reshape(-1)
    print(X.shape, X[0])
    print(Y.shape, Y[0]) 
    X_train, X_test, Y_train, Y_test = train_test_split(\
        X, Y, test_size=0.1, random_state=42)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    model = PySRRegressor(
        maxsize=20,
        niterations=40,  # < Increase me for better results
        binary_operators=["+", "*"],
        unary_operators=[
          "cos",
          "exp",
          "sin",
          "inv(x) = 1/x",
          # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        # ^ Custom loss function (julia syntax)
        verbosity=1
    )

    model.fit(X_train, Y_train)

    print("Best equations found:")
    print(model)
    print(model.get_best())
    print("Test R^2:", model.score(X_test, Y_test))
    print("Test MSE:", np.mean((model.predict(X_test) - Y_test) ** 2))
    print("Test MAE:", np.mean(np.abs(model.predict(X_test) - Y_test)))
    print("Some predictions:")
    for i in range(10):
        print(
            f"Input: {X_test[i]}, Prediction: {model.predict(X_test[i].reshape(1, -1))[0]}, True Value: {Y_test[i]}"
        )

    # Visualize predictions
    Y_pred = model.predict(X)
    Z_pred = Y_pred.reshape(z.shape)
    fig = plt.figure()
    fig.set_size_inches(12, 12)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, Z_pred, color='r', label='Predicted', alpha=0.5)
    ax.scatter(x, y, z, color='b', label='True', alpha=0.5)
    ax.set_xlabel('T (K)')
    ax.set_ylabel('v (quantum number)')
    ax.set_zlabel('log10 k (cm^3/s)')
    ax.legend()
    plt.savefig('predictions.png')
