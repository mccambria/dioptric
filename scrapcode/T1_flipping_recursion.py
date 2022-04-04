import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.special import comb

flip_prob = 0.1


def P(n, m):
    if n == 0:
        return 0
    elif n == 1 and m == 1:
        return 1
    else:
        return (1 - flip_prob) * P(n - 1, m - 1) + flip_prob * P(m - n, m - 1)


def N(m, i):

    options = np.arange(1, m + 1, 1)

    N_list = []
    for init_state in [True, False]:
        for combo in combinations(options, i):
            val = 0
            prev_ind = 0
            up = init_state
            for el in combo:
                if up:
                    val += el - prev_ind
                prev_ind = el
                up = not (up)
            if up:
                val += m - prev_ind
            N_list.append(val)
        N_counts = {val: N_list.count(val) for val in N_list}
    return N_counts


if __name__ == "__main__":

    # print(P(50, 100))

    # N = 15
    # n_vals = np.arange(0, N + 1, 1)
    # P_vals = [P(n, N) for n in n_vals]

    # plt.plot(n_vals, P_vals)
    # print(sum(P_vals))

    # plt.show(block=True)

    m = 10
    i = 4
    hist = N(m, i)
    probs = np.array(list(hist.values())) / (2 * comb(m, i))

    vals = hist.keys()
    plt.plot(vals, probs, "o")
    calc_probs = [
        (-((i + 1) / 2) * ((val - (m / 2)) ** 2) + m * i) / (2 * comb(m, i))
        for val in vals
    ]
    # calc_probs = [
    #     300
    #     * ((-(val - 1) / (m - 2)) ** 3 + (-(val - 1) / (m - 2)) ** 2)
    #     / comb(m, i)
    #     for val in vals
    # ]
    print(np.array(probs) - np.array(calc_probs))
    plt.plot(vals, calc_probs)

    plt.show(block=True)
