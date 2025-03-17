import numpy as np

def generate_simulation_data(K=10):
    A = np.random.randn(K)  # Random coefficients for sine terms
    B = np.random.randn(K)  # Random coefficients for cosine terms
    
    def generate_signal(t):
        # Generate the signal as a sum of sine and cosine terms
        terms = [(A[k] * np.sin(2 * np.pi * (k+1) * t) + B[k] * np.cos(2 * np.pi * (k+1) * t)) / (k+1)
                for k in range(K)]
        return sum(terms)
    return generate_signal

def generate_legendre_data(K=4):
    np.random.seed(123)  # Seed for reproducibility
    coeffs = np.random.randn(K)  # Random coefficients
    return np.polynomial.legendre.Legendre(coeffs, domain=[0, 1]), coeffs

def generate_legendre_inner_product_integrand(i, K=4):
    L_i = legendre_polynomial(i)  # Legendre polynomial of degree i
    f, c_i = generate_legendre_data(K)  # Function and its coefficients
    
    def integrand(t):
        # Integrand as the product of the Legendre polynomial and the function
        return L_i(t) * f(t)
    
    return integrand, c_i

def legendre_polynomial(i):
    coeffs = np.zeros(i + 1)  # Coefficients initialized to zero
    coeffs[i] = 1  # Set the coefficient of the i-th term to 1
    return np.polynomial.legendre.Legendre(coeffs, domain=[0, 1])

def product_of_legendre_polynomials(i, j):
    L_i = legendre_polynomial(i)  # Legendre polynomial of degree i
    L_j = legendre_polynomial(j)  # Legendre polynomial of degree j
    
    def integrand(t):
        # Integrand as the product of the two Legendre polynomials
        return L_i(t) * L_j(t)
    
    return integrand

def legendre_from_coefficients(coeffs):
    return np.polynomial.legendre.Legendre(coeffs, domain=[0, 1])

def estimate_riemann_integral(data):
    mask = np.isnan(data)  # Mask for NaN values
    if np.any(mask):
        # Replace NaNs with the previous valid value
        valid_idx = np.where(~mask, np.arange(len(data)), 0)
        np.maximum.accumulate(valid_idx, out=valid_idx)
        data = data[valid_idx]
    
    n = len(data)  # Number of data points
    integral = np.sum(data[:-1]) * 1 / n  # Riemann sum approximation
    return integral
