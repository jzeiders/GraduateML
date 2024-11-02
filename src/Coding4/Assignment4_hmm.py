import numpy as np

def BW_onestep(data, A, B, w):
    """
    Perform one iteration of the Baum-Welch algorithm (E-step and M-step).

    Parameters:
    -----------
    data: ndarray, shape (T,)
        Sequence of observations, each value in {0, 1, ..., mx-1}
    A: ndarray, shape (mz, mz)
        Current transition probability matrix
    B: ndarray, shape (mz, mx)
        Current emission probability matrix
    w: ndarray, shape (mz,)
        Initial state distribution (not updated)

    Returns:
    --------
    A_new: ndarray, shape (mz, mz)
        Updated transition probability matrix
    B_new: ndarray, shape (mz, mx)
        Updated emission probability matrix
    """
    T = len(data)
    mz, mx = B.shape
    
    print(mz,mx)

    # Initialize forward and backward probabilities
    alpha = np.zeros((T, mz))
    beta = np.zeros((T, mz))

    # Forward pass
    alpha[0] = w * B[:, data[0]]
    alpha[0] /= np.sum(alpha[0])  # Normalize to prevent underflow
    for t in range(1, T):
        for j in range(mz):
            alpha[t, j] = B[j, data[t]] * np.sum(alpha[t-1] * A[:, j])
        alpha[t] /= np.sum(alpha[t])  # Normalize

    # Backward pass
    beta[T-1] = np.ones(mz)
    beta[T-1] /= np.sum(beta[T-1])  # Normalize
    for t in range(T-2, -1, -1):
        for i in range(mz):
            beta[t, i] = np.sum(A[i, :] * B[:, data[t+1]] * beta[t+1, :])
        beta[t] /= np.sum(beta[t])  # Normalize

    # Compute gamma and xi
    gamma = np.zeros((T, mz))
    xi = np.zeros((T-1, mz, mz))
    for t in range(T):
        gamma[t] = alpha[t] * beta[t]
        gamma[t] /= np.sum(gamma[t])
    for t in range(T-1):
        denominator = np.sum(alpha[t] * A * B[:, data[t+1]].reshape(1, mz) * beta[t+1], axis=(0,1))
        for i in range(mz):
            for j in range(mz):
                xi[t, i, j] = alpha[t, i] * A[i, j] * B[j, data[t+1]] * beta[t+1, j]
        xi[t] /= np.sum(xi[t])

    # Update A
    A_new = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0).reshape(-1,1)

    # Update B
    B_new = np.zeros_like(B)
    for j in range(mz):
        for k in range(mx):
            B_new[j, k] = np.sum(gamma[data == k, j])
    B_new /= np.sum(gamma, axis=0).reshape(-1,1)

    return A_new, B_new

def myBW(data, mz, mx, initial_params, itmax):
    """
    Perform the Baum-Welch algorithm over multiple iterations.

    Parameters:
    -----------
    data: ndarray, shape (T,)
        Sequence of observations, each value in {0, 1, ..., mx-1}
    mz: int
        Number of hidden states
    mx: int
        Number of distinct observation symbols
    initial_params: dict
        Dictionary containing initial parameters:
            'A': transition matrix
            'B': emission matrix
            'w': initial state distribution
    itmax: int
        Maximum number of iterations

    Returns:
    --------
    A: ndarray, shape (mz, mz)
        Final transition probability matrix
    B: ndarray, shape (mz, mx)
        Final emission probability matrix
    ll: float
        Log-likelihood of the data given the model
    """
    A, B, w = initial_params['A'], initial_params['B'], initial_params['w']
    T = len(data)
    ll_history = []

    for iteration in range(itmax):
        A, B = BW_onestep(data, A, B, w)
        # Compute log-likelihood
        alpha = np.zeros((T, mz))
        alpha[0] = w * B[:, data[0]]
        alpha[0] /= np.sum(alpha[0])
        for t in range(0, T - 1):
            for j in range(mz):
                alpha[t, j] = B[j, data[t]] * np.sum(alpha[t-1] * A[:, j])
            alpha[t] /= np.sum(alpha[t])
        ll = np.sum(np.log(np.sum(alpha, axis=1)))
        ll_history.append(ll)
        print(f"Iteration {iteration+1}, Log-Likelihood: {ll}")

    return A, B, ll_history[-1]

def myViterbi(data, A, B, w):
    """
    Perform the Viterbi algorithm to find the most probable state sequence.

    Parameters:
    -----------
    data: ndarray, shape (T,)
        Sequence of observations, each value in {0, 1, ..., mx-1}
    A: ndarray, shape (mz, mz)
        Transition probability matrix
    B: ndarray, shape (mz, mx)
        Emission probability matrix
    w: ndarray, shape (mz,)
        Initial state distribution

    Returns:
    --------
    Z: ndarray, shape (T,)
        Most probable sequence of hidden states (0-based indexing)
    """
    T = len(data)
    mz = A.shape[0]

    # Initialize the log probabilities
    logA = np.log(A + 1e-16)  # Add small value to prevent log(0)
    logB = np.log(B + 1e-16)
    logw = np.log(w + 1e-16)

    # Initialize delta and psi
    delta = np.zeros((T, mz))
    psi = np.zeros((T, mz), dtype=int)

    # Initialization
    delta[0] = logw + logB[:, data[0]]
    psi[0] = 0

    # Recursion
    for t in range(1, T):
        for j in range(mz):
            temp = delta[t-1] + logA[:, j]
            psi[t, j] = np.argmax(temp)
            delta[t, j] = np.max(temp) + logB[j, data[t]]

    # Termination
    Z = np.zeros(T, dtype=int)
    Z[T-1] = np.argmax(delta[T-1])

    # Path backtracking
    for t in range(T-2, -1, -1):
        Z[t] = psi[t+1, Z[t+1]]

    return Z + 1  # Convert to 1-based indexing as per the benchmark

def load_data(file_path):
    """
    Load the observation sequence from a file.

    Parameters:
    -----------
    file_path: str
        Path to the data file

    Returns:
    --------
    data: ndarray, shape (T,)
        Sequence of observations as integers starting from 0
    """
    try:
        data = np.loadtxt(file_path, dtype=int)
        return data
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")

def load_Z(file_path):
    """
    Load the benchmark hidden state sequence from a file.

    Parameters:
    -----------
    file_path: str
        Path to the hidden state data file

    Returns:
    --------
    Z: ndarray, shape (T,)
        Sequence of hidden states as integers starting from 1
    """
    try:
        Z = np.loadtxt(file_path, dtype=int)
        return Z
    except Exception as e:
        raise ValueError(f"Error loading hidden states: {e}")

# Test functions
def test_BaumWelch(data, mz, mx, initial_params, itmax, expected_A=None, expected_B=None):
    """
    Test the Baum-Welch algorithm.

    Parameters:
    -----------
    data: ndarray, shape (T,)
        Sequence of observations
    mz: int
        Number of hidden states
    mx: int
        Number of observation symbols
    initial_params: dict
        Initial parameters
    itmax: int
        Number of iterations
    expected_A: ndarray, shape (mz, mz), optional
        Expected transition matrix for comparison
    expected_B: ndarray, shape (mz, mx), optional
        Expected emission matrix for comparison
    """
    A_final, B_final, ll_final = myBW(data, mz, mx, initial_params, itmax)
    print("\nFinal A matrix:")
    print(A_final)
    print("\nFinal B matrix:")
    print(B_final)
    print(f"\nFinal Log-Likelihood: {ll_final}")

    if expected_A is not None:
        print("\nDifference in A matrix:")
        print(A_final - expected_A)
    if expected_B is not None:
        print("\nDifference in B matrix:")
        print(B_final - expected_B)

def test_Viterbi(data, A, B, w, benchmark_Z=None):
    """
    Test the Viterbi algorithm.

    Parameters:
    -----------
    data: ndarray, shape (T,)
        Sequence of observations
    A: ndarray, shape (mz, mz)
        Transition probability matrix
    B: ndarray, shape (mz, mx)
        Emission probability matrix
    w: ndarray, shape (mz,)
        Initial state distribution
    benchmark_Z: ndarray, shape (T,), optional
        Benchmark hidden state sequence for comparison
    """
    Z = myViterbi(data, A, B, w)
    print("\nViterbi Most Probable State Sequence:")
    print(Z)

    if benchmark_Z is not None:
        if len(Z) != len(benchmark_Z):
            print("Benchmark Z length does not match.")
        else:
            matches = np.sum(Z == benchmark_Z)
            accuracy = matches / len(Z)
            print(f"\nViterbi Accuracy: {accuracy * 100:.2f}%")

def initialize_parameters(mz, mx):
    """
    Initialize the HMM parameters.

    Parameters:
    -----------
    mz: int
        Number of hidden states
    mx: int
        Number of observation symbols

    Returns:
    --------
    initial_params: dict
        Dictionary containing initial A, B, and w
    """
    # Initialize w uniformly
    w = np.full(mz, 1/mz)

    # Initialize A as per the test case
    A = np.array([[0.5, 0.5],
                  [0.5, 0.5]])

    # Initialize B as per the test case
    # Given B = [[1/9, 1/6, 3/6], [5/9, 2/6, 5/6]]
    B = np.array([[1/9, 1/6, 3/6],
                  [5/9, 2/6, 5/6]])

    return {'A': A, 'B': B, 'w': w}

# Main testing code
if __name__ == "__main__":
    # File paths (update these paths as per your directory structure)
    data_file = 'https://liangfgithub.github.io/Data/coding4_part2_data.txt'
    Z_benchmark_file = 'https://liangfgithub.github.io/Data/Coding4_part2_Z.txt'

    # Load data
    data = load_data(data_file) - 1  # Convert to 0-based indexing on states (TODO: Do this in a generic)
    print(data.shape)
    print(f"Loaded data sequence of length {len(data)}.")

    # Define parameters
    mx = 3  # Number of distinct observation symbols
    mz = 2  # Number of hidden states

    # Initialize parameters as specified
    initial_params = initialize_parameters(mz, mx)

    # Expected matrices after 100 iterations
    expected_A = np.array([[0.49793938, 0.50206062],
                           [0.44883431, 0.55116569]])
    expected_B = np.array([[0.22159897, 0.20266127, 0.57573976],
                           [0.34175148, 0.17866665, 0.47958186]])

    # Test Baum-Welch with G=2, itmax=100
    print("\n--- Running Baum-Welch Algorithm for 100 Iterations ---")
    test_BaumWelch(data, mz, mx, initial_params, itmax=100, expected_A=expected_A, expected_B=expected_B)

    # Run Viterbi
    print("\n--- Running Viterbi Algorithm ---")
    # Reload parameters after Baum-Welch
    A_final, B_final, ll_final = myBW(data, mz, mx, initial_params, itmax=100)
    Z = myViterbi(data, A_final, B_final, initial_params['w'])

    # Load benchmark Z
    benchmark_Z = load_Z(Z_benchmark_file)
    test_Viterbi(data, A_final, B_final, initial_params['w'], benchmark_Z)

    # Additional Test: Initialize B uniformly and run Baum-Welch for 20 and 100 iterations
    print("\n--- Testing with Uniform Initialization for B ---")
    # Initialize B uniformly
    B_uniform = np.full((mz, mx), 1/mx)
    initial_params_uniform = {'A': initial_params['A'], 'B': B_uniform, 'w': initial_params['w']}

    # Run for 20 iterations
    print("\nRunning Baum-Welch for 20 Iterations with Uniform B:")
    test_BaumWelch(data, mz, mx, initial_params_uniform, itmax=20)

    # Run for 100 iterations
    print("\nRunning Baum-Welch for 100 Iterations with Uniform B:")
    test_BaumWelch(data, mz, mx, initial_params_uniform, itmax=100)

    print("\n--- End of Testing ---")
