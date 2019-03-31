import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import scipy.io as io


def sigmoid(z):

    return 1 / 1 + np.exp(-z)


# def hypothesis(theta, X):
#     x_dot_theta_transpose = X.dot(theta.transpose())
#
#     return sigmoid(x_dot_theta_transpose)


def cost_function_j_(theta, X, Y, m, reg_lambda=0):

    theta = np.reshape(theta, (theta.size, 1))
    pred = sigmoid(np.dot(X, theta))
    J = np.mean(-Y * np.log(pred) - (1 - Y) * np.log(1 - pred))
    J = J + reg_lambda * sum(np.square(theta[1:])) / (2 * m)
    return J


def gradient_descent(X, Y, m, theta, alpha, iterations):
    cost_list = []

    for i in range(0, iterations):
        cost = cost_function_j_(theta, X, Y, m)
        print(cost)
        cost_list.append(cost)

        a = Y.reshape(1, m)
        b = hypothesis(theta, X)
        beta = hypothesis(theta, X) - Y.reshape(1, m)

        theta = theta - ((alpha / m) * X.transpose().dot(beta.transpose()).transpose())

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


def costFunction(theta, x, y, m, lamda=0):
    theta = np.reshape(theta, (theta.size, 1))
    pred = sigmoid(np.dot(x, theta))
    J = np.mean(-y * np.log(pred) - (1 - y) * np.log(1 - pred))
    J = J + lamda * sum(np.square(theta[1:])) / (2 * m)
    return J


def gradient(theta, x, y, m, lamda=0):
    theta = np.reshape(theta, (theta.size, 1))
    pred = sigmoid(np.dot(x, theta))
    grad = np.dot(x.T, pred - y) / m
    grad[1:] = grad[1:] + lamda * theta[1:] / m
    return grad.flatten()


# This function checks the probability of being number C and not C
def oneVsAll(theta, X, Y, lamda, C, miter, m):
    y = (Y == C).astype(int)
    theta = op.fmin_cg(costFunction, theta, fprime=gradient, args=(X, y, m, lamda), maxiter=miter, disp=0)
    return theta


def main():
    ex3 = io.loadmat("ex3data1.mat")

    m = ex3['X'].shape[0]

    X = np.column_stack((np.ones(m), ex3['X']))

    Y = ex3['y']

    # np.random.seed(1)
    # numbers = ex3['X'][np.random.choice(ex3['X'].shape[0], 25, replace=False)]
    # display_data(numbers)
    regularization_lambda = .1
    alpha = 0.1
    iterations = 5
    k = 10

    theta = np.zeros(X.shape[1])

    m = 499
    # result = op.fmin_cg(cost_function_j_, theta, args=(X[500:999], Y[500:999], m))
    # print(result)
    #
    # new_theta = gradient_descent(X[500:999], Y[500:999], m, theta, alpha, iterations)
    # print(new_theta)

    final_theta = np.zeros((k, X.shape[1]))

    # for i in range(0, k):
    #     final_theta[i] = gradient_descent(X[i * 500: i * 500 + 499],
    #                                       Y[i * 500: i * 500 + 499],
    #                                       m,
    #                                       theta,
    #                                       alpha,
    #                                       iterations)
    #
    #     final_theta[i] = op.fmin_cg(cost_function_j_, theta, args=(X[i * 500: i * 500 + 499],
    #                                                                      Y[i * 500: i * 500 + 499],
    #                                                                      m))

    for C in range(1, 11):
        final_theta[C % 10, :] = oneVsAll(theta, X, Y, regularization_lambda, C, iterations, m)
        print('Finished oneVsAll checking number: %d' % C)


if __name__ == "__main__":
    main()
