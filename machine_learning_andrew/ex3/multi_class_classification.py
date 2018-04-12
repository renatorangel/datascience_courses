import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import scipy.io as io


def normalize_data(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    sigma[sigma == 0] = 1.0
    x = (x-mu)/sigma
    #x = np.append(np.ones((x.shape[0], 1)), x, axis=1)
    return x, mu, sigma


def sigmoid(z):

    return 1.0 / (1.0 + np.exp(np.negative(z)))


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
        cost_list.append(cost)

        b = hypothesis(theta, X) - Y.reshape(1, m)

        theta[0] = theta.item(0) - (alpha / m) * np.sum(b)
        for i_theta in range(1, theta.shape[0]):

            theta[i_theta] = theta[i_theta] - ((alpha / m) * X[:, i_theta].dot(b.transpose())[0])

    plt.plot(range(0, iterations), cost_list)
    plt.ylabel('cost')
    plt.show()

    return theta


def display_data(numbers):
    f, axarr = plt.subplots(5, 5)

    for i in range(0, 5):
        for j in range(0, 5):
            num = numbers[j + 5 * i].reshape((20, 20)).transpose()
            axarr[i, j].imshow(num, cmap='gray_r', interpolation='nearest')
            axarr[i, j].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def main():
    ex3 = io.loadmat("ex3data1.mat")

    m = ex3['X'].shape[0]

    X = np.column_stack((np.ones(m), ex3['X']))#.astype(np.longdouble)

    Y = ex3['y'].astype(np.longdouble)

    np.random.seed(1)
    numbers = ex3['X'][np.random.choice(ex3['X'].shape[0], 25, replace=False)]
    display_data(numbers)

    alpha = 0.1
    iterations = 10000

    theta = np.repeat(0.1, X.shape[1]).astype(np.longdouble)

    m = 499
    result = op.fmin_cg(cost_function_j_, theta, args=(X[500:999], Y[500:999], m))
    print(result)
    # plot_graph(result, ex2)

    new_theta = gradient_descent(X[500:999], Y[500:999], m, theta, alpha, iterations)
    print(new_theta)
    # plot_graph(new_theta, ex2)


if __name__ == "__main__":
    main()
