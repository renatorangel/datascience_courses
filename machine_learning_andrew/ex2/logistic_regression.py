import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as op


def plot_graph(theta, ex2, mu, sigma):

    x1 = np.linspace(25.0, 100.0, 120)
    x2 = np.linspace(25.0, 100.0, 120)

    predicted_mat = np.empty((120, 120))
    for i, v1 in enumerate(x1):
        for j, v2 in enumerate(x2):
            v = np.array([v1, v2])
            predicted_mat[i, j] = predict(v, mu, sigma, theta)

    colors = ['r', 'b']
    lo = plt.scatter(ex2.loc[ex2.iloc[:, 2] == 0, 0],
                     ex2.loc[ex2.iloc[:, 2] == 0, 1],
                     marker='x',
                     color=colors[0])

    ll = plt.scatter(ex2.loc[ex2.iloc[:, 2] == 1, 0],
                     ex2.loc[ex2.iloc[:, 2] == 1, 1],
                     marker='o',
                     color=colors[1])

    plt.legend((lo, ll), ('Bad quality', 'Good quality'), scatterpoints=1)
    plt.contour(x1, x2, predicted_mat, [0.5])
    plt.ylabel('Test 2')
    plt.xlabel('Test 1')
    plt.show()


def sigmoid(z):

    return 1 / (1 + np.exp(np.negative(z)))


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
        # print(cost)
        cost_list.append(cost)

        b = hypothesis(theta, X) - Y.reshape(1, m)

        theta[0] = theta.item(0) - (alpha / m) * np.sum(b)
        theta[1] = theta.item(1) - (alpha / m) * X[:, 1].dot(b.transpose())[0]
        theta[2] = theta.item(2) - (alpha / m) * X[:, 2].dot(b.transpose())[0]

    plt.plot(range(0, iterations), cost_list)
    plt.ylabel('cost')
    plt.show()

    return theta


def normalize_data(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x = (x-mu)/sigma
    x = np.append(np.ones((x.shape[0], 1)), x, axis=1)
    return x, mu, sigma


def predict(array_grades, mu, sigma, model):
    x = (array_grades - mu) / sigma
    x = np.append(1, x)
    a = sigmoid(model.transpose().dot(x))
    return a


def main():
    ex2 = pd.read_csv("ex2data1.txt", header=None)

    m = len(ex2.index)

    X, mu, sigma = normalize_data(np.column_stack((ex2[0], ex2[1])))

    Y = ex2[2].as_matrix().reshape((m, 1))
    alpha = 0.003
    iterations = 300000
    theta = np.array([1.0, 1.0, 1.0])
    result_message = "Admission probability of {:.3%} from {}"

    result = op.fmin_bfgs(cost_function_j_, theta, args=(X, Y, m))

    print(result)
    print(result_message.format(predict(np.array([45, 85],),
                                        mu,
                                        sigma,
                                        result),
                                "fmin_bfgs"))

    plot_graph(result, ex2, mu, sigma)

    new_theta = gradient_descent(X, Y, m, theta, alpha, iterations)
    print(new_theta)
    print(result_message.format(predict(np.array([45, 85],),
                                        mu,
                                        sigma,
                                        new_theta),
                                "gradient_descent"))

    plot_graph(new_theta, ex2, mu, sigma)


if __name__ == "__main__":
    main()
