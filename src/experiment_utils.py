import numpy as np
from scipy.interpolate import BSpline

def generate_simulation_data(K=10):
    A = np.random.randn(K)
    B = np.random.randn(K)

    def generate_signal(t):
        terms = [(A[k] * np.sin(2 * np.pi * (k+1) * t) + B[k] * np.cos(2 * np.pi * (k+1) * t)) / (k+1)
                for k in range(K)]
        return sum(terms)
    return generate_signal

def generate_legendre_data(K=4):
    np.random.seed(123)
    coeffs = np.random.randn(K)
    return np.polynomial.legendre.Legendre(coeffs, domain=[0, 1]), coeffs

def generate_legendre_inner_product_integrand(i, K=4):
    L_i = legendre_polynomial(i)
    f, c_i = generate_legendre_data(K)
    def integrand(t):
        return L_i(t) * f(t)
    return integrand, c_i

def legendre_polynomial(i):
    coeffs = np.zeros(i + 1)
    coeffs[i] = 1
    return np.polynomial.legendre.Legendre(coeffs, domain=[0, 1])


def product_of_legendre_polynomials(i, j):
    L_i = legendre_polynomial(i)
    L_j = legendre_polynomial(j)
    def integrand(t):
        return L_i(t) * L_j(t)
    return integrand

def legendre_from_coefficients(coeffs):
    return np.polynomial.legendre.Legendre(coeffs, domain=[0, 1])

def estimate_riemann_integral(data):
    # Replace NaN values with the previous non-NaN value
    mask = np.isnan(data)
    if np.any(mask):
        valid_idx = np.where(~mask, np.arange(len(data)), 0)
        np.maximum.accumulate(valid_idx, out=valid_idx)
        data = data[valid_idx]
    
    n = len(data)
    if n < 2:
        return 0.0
    
    # Compute actual spacing between points
    dx = 1 / n
    
    # Compute Riemann sum using left-endpoint rule
    integral = np.sum(data[:-1]) * dx
    
    return integral