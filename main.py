import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import f

data_path = "./data/data.csv"
data = pd.read_csv(data_path)


def plane(a, b, c, x):
    return a*x+b*y+c


def LR_Model(data, predictors, outputs, graph=False):
    # Make the model
    x = data[predictors].values
    y = data[outputs].values
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=100)
    model = LinearRegression().fit(x_train, y_train)
    # Evaluate the model
    predicted_y = model.predict(x_test)
    mse = mean_squared_error(y_test, predicted_y)
    r_squared = model.score(x_train, y_train)
    n = len(x_train)
    p = x_train.shape[1]
    df = n - p - 1  # Denominator degrees of freedom
    # $F = \frac{R^2}{1-R^2} \frac{n-p-1}{p}$
    F = (r_squared/(1-r_squared)) * df/p
    p_value = f.sf(F, p, df)

    if graph:
        # Plot the data points
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x_train[:, 0], x_train[:, 1], y_train, color="blue", label="Training Data Points")
        ax.scatter(x_test[:, 0], x_test[:, 1], y_test, color="red", label="Testing Data Points")

        # Plot the prediction plane
        xs = np.linspace(0, ceil(max(x[:, 0])), 1000)
        ys = np.linspace(0, ceil(max(x[:, 1])), 1000)
        xs, ys = np.meshgrid(xs, ys)
        zs = model.intercept_ + model.coef_[0] * xs + model.coef_[1] * ys
        ax.plot_surface(xs, ys, zs, alpha=0.5, color='purple', label='Prediction Plane')

        # Make the graph pretty
        ax.set_xlabel("TB Cases")
        ax.set_ylabel("GDP")
        ax.set_zlabel("Life Expectancy")
        ax.set_title("Linear Regression of GDP and Tuberculosis Cases against Life Expectancy")
        ax.legend()
        plt.show()
    return {"MSE": mse, "F": F, "p": p_value, "r2": r_squared}


if __name__ == "__main__":
    model = LR_Model(data, ["TB Cases", "GDP"], "Life Expectancy", graph=True)
    print(
        f"Multiple Linear model between TB cases, GDP and life expectancy:\n\tr^2:\t\t{model["r2"]}\n\tMSE:\t\t{model["MSE"]}\n\tF:\t\t{model["F"]}\n\tp value:\t\t{model["p"]:.2e}\n\nThis means that a typical distance between a predicted value and the observed value is {sqrt(model["MSE"]):.2f}.")
