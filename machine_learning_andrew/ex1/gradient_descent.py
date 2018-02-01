import numpy as np
from ggplot import mtcars
import pandas as pd


def hypothesis(theta, X):
    theta_transpose = theta.transpose()

    return X.dot(theta_transpose)


def cost_function_j_(X, Y, m, theta):

    return np.sum(np.power(hypothesis(theta, X) - Y, 2)) / (2 * m)


def gradient_descent(X, Y, m, theta, alpha, iterations):

    for i in range(0, iterations):
        cost = cost_function_j_(X, Y, m, theta)
        print(cost)

        b = hypothesis(theta, X) - Y

        theta_0 = theta.item(0) - (alpha / m) * np.sum(b)
        theta_1 = theta.item(1) - (alpha / m) * X[:, 1].reshape((1, m)).dot(b)

        theta[0, 0] = theta_0
        theta[0, 1] = theta_1

    return theta


if __name__ == "__main__":

    ex1 = pd.read_csv("ex1data1.txt", header=None)
    m = len(ex1.index)
    X = np.column_stack((np.ones(m), ex1[0]))
    Y = ex1[1].as_matrix().reshape((m, 1))
    alpha = 0.01
    iterations = 1500
    theta = np.matrix([1, 1]).astype("float64")
    new_theta = gradient_descent(X, Y, m, theta, alpha, iterations)
    print(new_theta)

    m = len(mtcars.index)
    X = np.column_stack((np.ones(m), mtcars["mpg"]))
    Y = mtcars["hp"].as_matrix().reshape((32, 1))
    alpha = 0.001
    iterations = 1500
    theta = np.matrix([1, 1]).astype("float64")
    new_theta = gradient_descent(X, Y, m, theta, alpha, iterations)
    print(new_theta)
