import numpy as np

def generate_simulation_data(K=10):
    A = np.random.randn(K)
    B = np.random.randn(K)

    def generate_signal(t):
        terms = [(A[k] * np.sin(2 * np.pi * (k+1) * t) + B[k] * np.cos(2 * np.pi * (k+1) * t)) / (k+1)
                for k in range(K)]
        return sum(terms)
    return generate_signal, A, B

def generate_legendre_basis_expansion(K=4):
    np.random.seed(123)
    coeffs = np.random.randn(K)
    return np.polynomial.legendre.Legendre(coeffs, domain=[0, 1]), coeffs

def get_legendre_polynomial(i, K=4):
    coeffs = np.zeros(K)
    coeffs[i] = 1
    return np.polynomial.legendre.Legendre(coeffs, domain=[0, 1])

def multiply_functions(f, g):
    def product(t):
        return f(t) * g(t)
    return product