import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as op


# got this code from here: https://github.com/rragundez/coursera-machine-learning-AndrewNg-Python
def create_polynomial_features(x, pol_deg):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    X = np.ones((x1.size, 1))
    for n in range(1, pol_deg + 1):
        for m in range(n + 1):
            X = np.append(X, x1 ** (n - m) * x2 ** m, axis=1)
    return X


def get_boundary_values(x, theta):
    return (0.5 - theta[0] - theta[2] * x) / theta[1]


def plot_graph(formula, X, theta, ex2):
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

    a = 1 / (1 + np.exp(np.negative(z)))

    return a


def hypothesis(theta, X):
    x_dot_theta_transpose = X.dot(theta.transpose())

    return sigmoid(x_dot_theta_transpose)


def cost_function_j_(theta, X, Y, m):
    h_theta = hypothesis(theta, X)

    y_transpose_m_log_h_theta = np.multiply(-Y.transpose(), np.log(h_theta))

    one_minus_y_transpose = np.multiply(1 - Y.transpose(), np.log(1 - h_theta))

    return np.sum(y_transpose_m_log_h_theta - one_minus_y_transpose) / m


def gradient_descent(X, Y, m, theta, alpha, iterations):
    cost_list = []

    for i in range(0, iterations):
        cost = cost_function_j_(theta, X, Y, m)
        print(cost)
        cost_list.append(cost)

        b = hypothesis(theta, X) - Y.reshape(1, m)

        theta[0] = theta.item(0) - (alpha / m) * np.sum(b)
        for index_theta in range(1, theta.shape[0]):

            theta[index_theta] = theta[index_theta] - (alpha / m) * X[:, index_theta].dot(b.transpose())[0]

    plt.plot(range(0, iterations), cost_list)
    plt.ylabel('cost')
    plt.show()

    return theta


def main():
    ex2 = pd.read_csv("ex2data2.txt", header=None)

    X = create_polynomial_features(np.column_stack((ex2[0], ex2[1])), 6)
    m = len(ex2.index)

    Y = ex2[2].as_matrix().reshape((m, 1))
    alpha = 0.001
    iterations = 100000

    theta = np.repeat(0.1, X.shape[1])
    print(theta.shape)
    result = op.fmin_bfgs(cost_function_j_, theta, args=(X, Y, m))

    print(result)

    plot_graph(get_boundary_values, X, result, ex2)

    new_theta = gradient_descent(X, Y, m, theta, alpha, iterations)
    print(new_theta)

    plot_graph(get_boundary_values, X, new_theta, ex2)


if __name__ == "__main__":
    main()
