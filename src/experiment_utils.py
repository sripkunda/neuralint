import numpy as np

def generate_simulation_data(K=10):
    A = np.random.randn(K)
    B = np.random.randn(K)

    def generate_signal(t):
        terms = [(A[k] * np.sin(2 * np.pi * (k+1) * t) + B[k] * np.cos(2 * np.pi * (k+1) * t)) / (k+1)
                for k in range(K)]
        return sum(terms)
    return generate_signal