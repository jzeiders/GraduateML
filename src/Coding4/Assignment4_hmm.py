import numpy as np
import requests
from io import StringIO

def forward_pass(data, w, A, B):
    """
    Compute forward probabilities alpha(t,j)
    
    Args:
        data: T-by-1 observation sequence (1D array)
        w: Initial state distribution (mz,)
        A: Transition matrix (mz-by-mz)
        B: Emission matrix (mz-by-mx)
    
    Returns:
        alpha: Forward probabilities (T-by-mz)
    """
    T = len(data)
    mz = len(w)
    alpha = np.zeros((T, mz))
    
    # Initialize first time step
    alpha[0, :] = w * B[:, data[0]]
    
    # Forward recursion
    for t in range(1, T):
        alpha[t, :] = np.dot(alpha[t-1, :], A) * B[:, data[t]]
    
    return alpha

def backward_pass(data, A, B):
    """
    Compute backward probabilities beta(t,j)
    
    Args:
        data: T-by-1 observation sequence (1D array)
        A: Transition matrix (mz-by-mz)
        B: Emission matrix (mz-by-mx)
        c: Scaling factors from forward pass (T,)
    
    Returns:
        beta: Backward probabilities (T-by-mz)
    """
    T = len(data)
    mz = A.shape[0]
    beta = np.zeros((T, mz))
    
    # Initialize last time step
    beta[-1, :] = 1
    
    # Backward recursion
    for t in range(T-2, -1, -1):
        beta[t, :] = np.dot(A, (B[:, data[t+1]] * beta[t+1, :]))
    
    return beta

def BW_onestep(data, w, A, B):
    """
    One step of the Baum-Welch algorithm (E-step + M-step)
    
    Args:
        data: T-by-1 observation sequence (1D array)
        w: Initial state distribution (mz,)
        A: Current transition matrix (mz-by-mz)
        B: Current emission matrix (mz-by-mx)
    
    Returns:
        A_new: Updated transition matrix
        B_new: Updated emission matrix
    """
    T = len(data)
    mz = A.shape[0]
    mx = B.shape[1]
    
    # E-step: Compute forward and backward probabilities
    alpha = forward_pass(data, w, A, B)
    beta = backward_pass(data, A, B)
    
    # Compute xi(t,i,j) = P(z_t=i, z_{t+1}=j | x_{1:T})
    xi = np.zeros((T-1, mz, mz))
    for t in range(T-1):
        denominator = np.dot(alpha[t, :], A * B[:, data[t+1]].reshape(-1,1)) @ beta[t+1, :]
        xi[t, :, :] = (alpha[t, :, np.newaxis] * A * B[:, data[t+1]].reshape(-1,1) * beta[t+1, :]) / denominator
    
    # Compute gamma(t,j) = P(z_t=j | x_{1:T})
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)  # Explicit normalization
    
    # M-step: Update parameters
    # Update A
    A_new = np.sum(xi, axis=0) / np.sum(gamma[:-1, :], axis=0)[:, np.newaxis]
    
    # Update B
    B_new = np.zeros_like(B)
    for j in range(mz):
        for k in range(mx):
            B_new[j, k] = np.sum(gamma[data == k, j])
        B_new[j, :] /= np.sum(gamma[:, j])
    
    return A_new, B_new

def myBW(data, mz, mx, initial_params, itmax):
    """
    Main Baum-Welch algorithm with specified initial parameters
    
    Args:
        data: T-by-1 observation sequence (1D array)
        mz: Number of hidden states
        mx: Number of observation symbols
        initial_params: Dictionary containing initial parameters:
            'w': Initial state distribution (mz,)
            'A': Initial transition matrix (mz-by-mz)
            'B': Initial emission matrix (mz-by-mx)
        itmax: Maximum number of iterations
    
    Returns:
        A_final: Final transition matrix
        B_final: Final emission matrix
        ll_final: Final log likelihood
    """
    # Ensure data is a 1D array
    data = np.asarray(data).flatten()
    
    # Extract initial parameters
    w = initial_params['w']
    A = initial_params['A']
    B = initial_params['B']
    
    
    for iteration in range(itmax):
        # Compute log likelihood
        alpha = forward_pass(data, w, A, B)
        current_ll = np.sum(np.log(alpha[:, -1]))
        
        
        # Update parameters
        A, B = BW_onestep(data, w, A, B)
        
        # Debugging: Print row sums to ensure they sum to 1
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"\nIteration {iteration + 1}")
            print("Row sums of A:", A.sum(axis=1))
            print("Row sums of B:", B.sum(axis=1))
            print(f"Log-Likelihood: {current_ll}")
    
    return A, B, current_ll

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

def load_data(file_path_or_url):
    """
    Load the observation sequence from a file or URL.

    Parameters:
    -----------
    file_path_or_url: str
        Path to the data file or URL

    Returns:
    --------
    data: ndarray, shape (T,)
        Sequence of observations as integers starting from 0
    """
    try:
        if file_path_or_url.startswith('http://') or file_path_or_url.startswith('https://'):
            response = requests.get(file_path_or_url)
            response.raise_for_status()  # Ensure we notice bad responses
            data = np.loadtxt(StringIO(response.text), dtype=int)
        else:
            data = np.loadtxt(file_path_or_url, dtype=int)
        return data
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")

def load_Z(file_path_or_url):
    """
    Load the benchmark hidden state sequence from a file or URL.

    Parameters:
    -----------
    file_path_or_url: str
        Path to the hidden state data file or URL

    Returns:
    --------
    Z: ndarray, shape (T,)
        Sequence of hidden states as integers starting from 1
    """
    try:
        if file_path_or_url.startswith('http://') or file_path_or_url.startswith('https://'):
            response = requests.get(file_path_or_url)
            response.raise_for_status()
            Z = np.loadtxt(StringIO(response.text), dtype=int)
        else:
            Z = np.loadtxt(file_path_or_url, dtype=int)
        return Z
    except Exception as e:
        raise ValueError(f"Error loading hidden states: {e}")

def validate_data(data, mx):
    """
    Validate that all observation symbols are within the range [0, mx-1].

    Parameters:
    -----------
    data: ndarray, shape (T,)
        Sequence of observations
    mx: int
        Number of observation symbols

    Raises:
    -------
    ValueError: If any observation symbol is out of range.
    """
    if not np.all((data >= 0) & (data < mx)):
        raise ValueError(f"Data contains symbols outside the range [0, {mx-1}].")

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
    print("Row sums of A:", A_final.sum(axis=1))
    
    print("\nFinal B matrix:")
    print(B_final)
    print("Row sums of B:", B_final.sum(axis=1))
    
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
    B = np.array([[1/9, 1/6, 3/6],
                  [5/9, 2/6, 5/6]])

    # Normalize the rows of B to ensure they sum to 1
    B /= B.sum(axis=1, keepdims=True)

    return {'A': A, 'B': B, 'w': w}

# Main testing code
if __name__ == "__main__":
    # File paths (update these paths as per your directory structure)
    data_file = 'https://liangfgithub.github.io/Data/coding4_part2_data.txt'
    Z_benchmark_file = 'https://liangfgithub.github.io/Data/Coding4_part2_Z.txt'

    # Load data
    data = load_data(data_file) - 1  # Convert to 0-based indexing
    print(f"Loaded data sequence of length {len(data)}.")
    
    # Define parameters
    mx = 3  # Number of distinct observation symbols
    mz = 2  # Number of hidden states

    # Validate data
    validate_data(data, mx)

    # Initialize parameters as specified
    initial_params = initialize_parameters(mz, mx)

    # Expected matrices after 100 iterations (for comparison, optional)
    expected_A = np.array([[0.49793938, 0.50206062],
                           [0.44883431, 0.55116569]])
    expected_B = np.array([[0.22159897, 0.20266127, 0.57573976],
                           [0.34175148, 0.17866665, 0.47958186]])

    # Test Baum-Welch with itmax=100 for demonstration (adjust as needed)
    print("\n--- Running Baum-Welch Algorithm for 100 Iterations ---")
    test_BaumWelch(data, mz, mx, initial_params, itmax=100, expected_A=expected_A, expected_B=expected_B)

    # Uncomment and modify the following lines to run Viterbi and additional tests with actual data files

    # # Run Viterbi
    # print("\n--- Running Viterbi Algorithm ---")
    # # Reload parameters after Baum-Welch
    # A_final, B_final, ll_final = myBW(data, mz, mx, initial_params, itmax=100)
    # Z = myViterbi(data, A_final, B_final, initial_params['w'])

    # # Load benchmark Z
    # benchmark_Z = load_Z(Z_benchmark_file)
    # test_Viterbi(data, A_final, B_final, initial_params['w'], benchmark_Z)

    # # Additional Test: Initialize B uniformly and run Baum-Welch for 20 and 100 iterations
    # print("\n--- Testing with Uniform Initialization for B ---")
    # # Initialize B uniformly
    # B_uniform = np.full((mz, mx), 1/mx)
    # initial_params_uniform = {'A': initial_params['A'], 'B': B_uniform, 'w': initial_params['w']}

    # # Run for 20 iterations
    # print("\nRunning Baum-Welch for 20 Iterations with Uniform B:")
    # test_BaumWelch(data, mz, mx, initial_params_uniform, itmax=20)

    # # Run for 100 iterations
    # print("\nRunning Baum-Welch for 100 Iterations with Uniform B:")
    # test_BaumWelch(data, mz, mx, initial_params_uniform, itmax=100)

    # print("\n--- End of Testing ---")
