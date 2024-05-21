import numpy as np
from scipy.optimize import linprog


def calculate_probabilities_lp(token_amounts, target_avg, epsilon=0.01, min_prob=0.01):
    n = len(token_amounts)

    # Coefficients for the objective function (dummy, as we only care about constraints)
    c = np.zeros(n)

    # Coefficients matrix for equality constraints
    A_eq = [np.ones(n), token_amounts]

    # Right-hand side of equality constraints
    b_eq = [1, target_avg]

    # Coefficients matrix for inequality constraints (p_i >= p_{i+1} + epsilon)
    A_ub = np.zeros((n-1, n))
    for i in range(n-1):
        A_ub[i, i] = -1
        A_ub[i, i+1] = 1

    # Right-hand side of inequality constraints
    b_ub = np.full(n-1, -epsilon)

    # Bounds for each probability (min_prob <= p_i <= 1)
    bounds = [(min_prob, 1) for _ in range(n)]

    # Solve the linear programming problem
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub,
                     b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        probabilities = result.x
    else:
        raise ValueError("Linear programming did not converge")

    return probabilities


# Example usage
token_amounts = [200, 400, 600, 1000, 2000, 8000, 10000]
target_avg = 1000
min_prob = 0.001
probabilities = calculate_probabilities_lp(
    token_amounts, target_avg, min_prob=min_prob)

actual_avg = sum([token_amounts[i] * probabilities[i]
                 for i in range(len(token_amounts))])


def format_probability(p):
    return f"{(p * 100):.2f}%"


for i, amount in enumerate(token_amounts):
    print(
        f"Token {i+1}: ${amount}, Probability: {format_probability(probabilities[i])}")
print("EV: ", actual_avg)
