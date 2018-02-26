import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def my_formula(x, theta):
    return (0.5 - theta[0, 0] - theta[0, 2] * x) / theta[0, 1]


def graph(formula, X, theta, ex2):
    boundary_values = formula(X[:, 2], theta)

    colors = ['r', 'b']
    lo = plt.scatter(ex2.loc[ex2.iloc[:, 2] == 0, 0], ex2.loc[ex2.iloc[:, 2] == 0, 1], marker='x', color=colors[0])
    ll = plt.scatter(ex2.loc[ex2.iloc[:, 2] == 1, 0], ex2.loc[ex2.iloc[:, 2] == 1, 1], marker='o', color=colors[1])
    plt.legend((lo, ll), ('Not admitted', 'Admitted'), scatterpoints=1)

    plt.plot(X[:, 2], boundary_values)
    plt.ylabel('exam2')
    plt.xlabel('exam1')
    plt.show()


def sigmoid(z):

    b = np.exp(np.negative(z), dtype=np.longdouble)
    c = np.add(np.ones(z.shape[0]).astype(np.longdouble).reshape(z.shape[0], 1), b, dtype=np.longdouble)

    return 1 / c


def hypothesis(theta, X):

    return sigmoid(X.dot(theta.transpose()))


def cost_function_j_(X, Y, m, theta):
    h_theta = hypothesis(theta, X)

    a = np.multiply(np.negative(Y), np.log(h_theta))

    b = np.multiply(1 - Y, np.log(1 - h_theta))

    return np.sum(a - b) / m


def gradient_descent(X, Y, m, theta, alpha, iterations, ex2):
    cost_list = []

    for i in range(0, iterations):
        cost = cost_function_j_(X, Y, m, theta)
        print(cost)
        cost_list.append(cost)

        b = hypothesis(theta, X) - Y

        theta_0 = theta.item(0) - (alpha / m) * np.sum(b)
        theta_1 = theta.item(1) - (alpha / m) * X[:, 1].reshape((1, m)).dot(b)
        theta_2 = theta.item(2) - (alpha / m) * X[:, 2].reshape((1, m)).dot(b)

        theta[0, 0] = theta_0
        theta[0, 1] = theta_1
        theta[0, 2] = theta_2

    plt.plot(range(0, iterations), cost_list)
    plt.ylabel('cost')
    plt.show()

    graph(my_formula, X, theta, ex2)
    return theta


def main():

    ex2 = pd.read_csv("ex2data1.txt", header=None)

    m = len(ex2.index)
    X = np.column_stack((np.ones(m), ex2[0], ex2[1])).astype(np.longdouble)
    Y = ex2[2].as_matrix().reshape((m, 1)).astype(np.longdouble)
    alpha = 0.001
    iterations = 150000
    theta = np.matrix([.1, .1, .1]).astype(np.longdouble)
    new_theta = gradient_descent(X, Y, m, theta, alpha, iterations, ex2)
    print(new_theta)


if __name__ == "__main__":
    main()
