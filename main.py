import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm
from sympy import Symbol, integrate, exp, sqrt, pi, Abs

# Define parameters for the Gaussian distribution
mu = 0
sigma = 2

# Time window
a = 0
b = 5

x = Symbol('x')
u = Symbol('u')


def gaussian_pdf(x):
    norm_factor = norm.cdf((b - mu) / sigma) - norm.cdf((a - mu) / sigma)
    if norm_factor <= 0:
        raise ValueError("Normalization factor is zero or negative.")
    return (1 / (sigma * sqrt(2 * np.pi))) * exp(-0.5 * ((x - mu) / sigma) ** 2) / norm_factor


def detection_function(u, x):
    return Abs(u - x)


def get_total_penalty_func(num_of_points):
    t = [Symbol("t" + str(i)) for i in range(1, num_of_points + 1)]
    t.insert(0, a)  # Start at a
    t.append(b)  # End at b
    penalty_func = []

    for idx in range(num_of_points + 1):  # Include the last point
        penalty_func.append(integrate(detection_function(t[idx], x), (x, t[idx], t[idx + 1])))

    return penalty_func


def get_expected_penalty_func(func):
    integrals = []
    t = ([Symbol("t" + str(i)) for i in range(1, len(func))])
    t.insert(0, a)  # Start from a
    t.append(b)  # End with b

    for idx, term in enumerate(func):
        integral_value = integrate(term * gaussian_pdf(x), (x, t[idx], t[idx + 1]))
        print(f"Penalty for interval [{t[idx]}, {t[idx + 1]}]: {integral_value}")
        integrals.append(integral_value)

    return integrals


def objective(t_values):
    keys = ["t" + str(i) for i in range(1, num_points + 1)]
    subs = {key: value for key, value in zip(keys, t_values)}
    total_penalty = sum([penalty.evalf(subs=subs) for penalty in total_expected_penalty])
    print(f"Current total penalty: {total_penalty}")
    return total_penalty


num_points_list = [1, 2, 3, 10]
file_name = "gaussian"

for num_points in num_points_list:
    penalty_func = get_total_penalty_func(num_points)
    total_expected_penalty = get_expected_penalty_func(penalty_func)

    t0 = np.linspace(a, b, num_points)

    # Bounds for t: each t should be within [t_min, t_max]
    bounds = [(a, b) for _ in range(num_points)]

    # Constraints to ensure t[i] < t[i+1]
    constraints = [{'type': 'ineq', 'fun': lambda t: t[i + 1] - t[i]} for i in range(num_points - 1)]

    solution = minimize(objective, t0, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_points = solution.x
    print("Final value " + str(objective(optimal_points)))

    print('Optimal monitoring points:')
    for i, t in enumerate(optimal_points, 1):
        print(f't{i} = {t}')

    # Plotting the Gaussian PDF and optimal monitoring points
    x_vals = np.linspace(a, b, 1000)
    y_vals = [float(gaussian_pdf(val)) for val in x_vals]

    plt.plot(x_vals, y_vals, label='Gaussian PDF')
    plt.scatter(optimal_points, [float(gaussian_pdf(val)) for val in optimal_points], color='red', zorder=5)
    plt.title('Optimal Monitoring Points on Gaussian Distribution')
    plt.xlabel('Time')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.savefig(str(file_name) + "_" + str(mu).replace(".", ",") + "mu" + "_" +
                str(sigma).replace(".", ",") + "sigma" + "_" + str(num_points) + "pts.png")
    plt.show()
