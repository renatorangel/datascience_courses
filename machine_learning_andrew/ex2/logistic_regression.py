import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def hypothesis_sigmoid(theta, X):
    pass


def plot_df():
    colors = ['r', 'b']
    lo = plt.scatter(ex2.loc[ex2.iloc[:, 2] == 0, 0], ex2.loc[ex2.iloc[:, 2] == 0, 1], marker='x', color=colors[0])
    ll = plt.scatter(ex2.loc[ex2.iloc[:, 2] == 1, 0], ex2.loc[ex2.iloc[:, 2] == 1, 1], marker='o', color=colors[1])
    plt.legend((lo, ll), ('Not admitted', 'Admitted'), scatterpoints=1)
    plt.show()


if __name__ == "__main__":

    ex2 = pd.read_csv("ex2data1.txt", header=None)

    plot_df()
