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


def line(a, b, x):
    """
    Compute the value of a line using the linear equation

    $$ y = ax + b $$

    Parameters:
    -----------
    a : float
        The slope of the line.
    b : float
        The y-intercept of the line.
    x : float or array-like
        The value or array of values at which to evaluate the line.

    Returns:
    --------
    float
        The computed y value corresponding to the input x.
    """
    return a*x+b


def LR_Model(data, h1, h2, graph=False):
    """
    Build and evaluate a linear regression model.

    This function splits the input dataset into training and testing subsets, fits a linear regression 
    model using the specified independent and dependent variables, and computes key evaluation metrics 
    including the Mean Squared Error (MSE), coefficient of determination R^2, the F-statistic, 
    and the corresponding p-value. Optionally, it plots the regression line along with the data 
    points and predicted values.

    The F-statistic is computed by the formula:

    $F = \\frac{R^2}{1-R^2} \\frac{n-p-1}{p}$

    where n is the number of training samples and p=1 is the number of predictors.

    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset containing the variables.
    h1 : str
        The column name in data to be used as the independent variable.
    h2 : str
        The column name in data to be used as the dependent variable.
    graph : bool, optional
        If True, displays a plot of the regression line along with the data points and the predicted points.
        Default is False.

    Returns:
    --------
    dict
        A dictionary containing the following keys:
        - "MSE": The Mean Squared Error of the predictions.
        - "F": The computed F-statistic.
        - "p": The p-value corresponding to the F-statistic.
        - "r2": The coefficient of determination R^2 of the model on the training data.
    """
    x = data[h1].values.reshape(-1, 1)
    y = data[h2].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=100)

    model = LinearRegression().fit(x_train, y_train)

    predicted_y = model.predict(x_test)

    mse = mean_squared_error(y_test, predicted_y)
    r_squared = model.score(x_train, y_train)
    n = len(x_train)
    p = 1
    df = n - p - 1  # Denominator degrees of freedom
    # $F = \frac{R^2}{1-R^2} \frac{n-p-1}{p}$
    F = (r_squared/(1-r_squared)) * df/p
    p_value = f.sf(F, p, df)

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

    return {"MSE": mse, "F": F, "p": p_value, "r2": r_squared}


if __name__ == "__main__":
    model = LR_Model(data, "TB Cases", "Life Expectancy")
    print(f"Linear model between TB cases and life expectancy:\n\tr^2:\t\t{model["r2"]}\n\tMSE:\t\t{model["MSE"]}\n\tF:\t\t{model["F"]}\n\tp value:\t\t{model["p"]:.2e}\n\nHence this is not a great model but its actually by no means bad because it means that a typical distance between a predicted value and the observed value is {sqrt(model["MSE"]):.2f}.")
