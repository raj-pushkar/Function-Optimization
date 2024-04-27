import numpy as np
import matplotlib.pyplot as plt


def evolution_Strategy(f, population_size, sigma, lr, initial_params, num_iters):

    num_params = len(initial_params)
    rewards_per_iteration = np.zeros(num_iters)

    params = initial_params

    for t in range(num_iters):
        N = np.random.randn(population_size, num_params)
        R = np.zeros(population_size)

        for i in range(population_size):
            params_try = params + sigma * N[i]
            R[i] = f(params_try)

        m = R.mean()
        s = R.std()
        A = (R - m) / s
        rewards_per_iteration[t] = m
        params = params + lr/(population_size*sigma) * np.dot(N.T, A)

    return params, rewards_per_iteration


def rewardFunction(params):
    x = params[0]
    y = params[1]
    z = params[2]
    return -(x**2 + 0.1 * (y - 1)**2 + 0.5 * (z + 2)**2)


if __name__ == '__main__':
    best_params, rewards = evolution_Strategy(
        f=rewardFunction, population_size=50, sigma=0.1, lr=1e-3, initial_params=np.random.randn(3), num_iters=500)

    plt.plot(rewards)
    plt.show()

    print("Final params: ", best_params)
