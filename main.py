import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data_path = "./data/data.csv"
data = pd.read_csv(data_path)


def line(a, b, x):
    return a*x+b


def LR_Model(data, h1, h2, graph=False):
    x = data[h1].values.reshape(-1, 1)
    y = data[h2].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=100)

    model = LinearRegression()
    model.fit(x_train, y_train)

    predicted_y = model.predict(x_test)
    mse = mean_squared_error(y_test, predicted_y)

    if graph:
        xs = np.linspace(0, ceil(max(x)[0]), 1000)
        plt.plot(xs, line(model.coef_[0], model.intercept_, xs), marker='o',
                 linewidth=1, markersize=0, color="purple", label="Linear Regression model")
        plt.scatter(x, y, color="blue", s=2, label="Data Points")
        plt.scatter(x_test, predicted_y, color="red",
                    s=3, label="Predicted Points")
        plt.xlabel("TB Cases")
        plt.ylabel("Life Expectancy")
        plt.title("Linear Regression of Tuberculosis Cases against Life Expectancy")
        plt.legend()
        plt.show()

    return {"MSE": mse}


if __name__ == "__main__":
    model = LR_Model(data, "TB Cases", "Life Expectancy", True)
    print(
        f"MSE of the linear model between TB cases and life expectancy: {model["MSE"]}\nHence this is not a very good model (but that wasn't really the point). This data is perhaps not the best candidate for a linear regression, by visual inspection, because there are so many countries with no TB cases but wildly different life expectancy.")
