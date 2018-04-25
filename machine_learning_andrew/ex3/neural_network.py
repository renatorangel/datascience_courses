import scipy.io as io
import numpy as np


def main():
    ex3_weights = io.loadmat("ex3weights.mat")
    theta1 = ex3_weights["Theta1"]
    theta2 = ex3_weights["Theta2"]
    print(ex3_weights)


if __name__ == "__main__":
    main()
