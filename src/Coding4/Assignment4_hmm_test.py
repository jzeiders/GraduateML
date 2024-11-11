from io import StringIO
import unittest
import numpy as np

from numpy.testing import assert_array_almost_equal, assert_array_less


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
        # np.dot(alpha[t-1, :], A) gives a (mz,) shape
        # B[:, data[t]] also gives a (mz,) shape
        # We need to ensure the shapes align for element-wise multiplication
        alpha[t, :] = np.dot(alpha[t - 1, :], A) * B[:, data[t]].T

    return alpha


class TestHMMForward(unittest.TestCase):
    def setUp(self):
        """Set up some common HMM parameters that will be used across tests"""
        # Simple 2-state, 2-observation HMM
        self.w_simple = np.array([0.6, 0.4])
        self.A_simple = np.array([[0.7, 0.3], [0.4, 0.6]])
        self.B_simple = np.array([[0.5, 0.5], [0.2, 0.8]])
        self.data_simple = np.array([0, 1, 0])

    def test_basic_properties(self):
        """Test basic mathematical properties of forward probabilities"""
        alpha = forward_pass(
            self.data_simple, self.w_simple, self.A_simple, self.B_simple
        )

        # Test non-negative probabilities
        self.assertTrue(np.all(alpha >= 0), "Found negative probabilities")

        # Test sum of probabilities <= 1
        self.assertTrue(
            np.all(np.sum(alpha, axis=1) <= 1 + 1e-10), "Probabilities sum > 1"
        )

        # Test first time step calculation
        expected_first = self.w_simple * self.B_simple[:, self.data_simple[0]]
        self.assertTrue(
            np.allclose(alpha[0], expected_first), "First step calculation incorrect"
        )

    def test_deterministic_sequence(self):
        """Test with a nearly deterministic HMM"""
        w = np.array([1.0, 0.0])  # Start in state 0
        A = np.array([[0.9, 0.1], [0.1, 0.9]])  # Mostly stay in same state
        B = np.array(
            [[0.9, 0.1], [0.1, 0.9]]  # State 0 mostly emits 0
        )  # State 1 mostly emits 1
        data = np.array([0, 0, 0])

        alpha = forward_pass(data, w, A, B)
        self.assertTrue(
            np.all(alpha[:, 0] > alpha[:, 1]),
            "Expected state 0 to have higher probability",
        )

    def test_edge_cases(self):
        """Test edge cases and corner conditions"""
        # Test single observation
        w = np.array([0.5, 0.5])
        A = np.array([[0.5, 0.5], [0.5, 0.5]])
        B = np.array([[0.5, 0.5], [0.5, 0.5]])
        data = np.array([0])

        alpha = forward_pass(data, w, A, B)
        self.assertEqual(alpha.shape, (1, 2), "Incorrect shape for single observation")

        # Test uniform distributions
        data = np.array([0, 1, 0])
        alpha = forward_pass(data, w, A, B)
        self.assertTrue(
            np.allclose(alpha[:, 0], alpha[:, 1]),
            "Expected equal probabilities with uniform distributions",
        )

    def test_known_sequence(self):
        """Test with manually calculated expected values"""
        w = np.array([0.8, 0.2])
        A = np.array([[0.6, 0.4], [0.3, 0.7]])
        B = np.array([[0.7, 0.3], [0.1, 0.9]])
        data = np.array([0, 1])

        alpha = forward_pass(data, w, A, B)

        # Check first time step
        expected_t0 = w * B[:, data[0]]
        self.assertTrue(np.allclose(alpha[0], expected_t0), "First time step incorrect")

        # Check second time step
        expected_t1 = B[:, data[1]] * (A.T @ alpha[0])
        self.assertTrue(
            np.allclose(alpha[1], expected_t1), "Second time step incorrect"
        )

    def test_numerical_stability(self):
        """Test with very small probabilities"""
        w = np.array([0.99, 0.01])
        A = np.array([[0.99, 0.01], [0.01, 0.99]])
        B = np.array([[0.99, 0.01], [0.01, 0.99]])
        data = np.array([0, 0, 0, 0, 0])  # Long sequence to test stability

        alpha = forward_pass(data, w, A, B)
        self.assertTrue(
            np.all(np.isfinite(alpha)),
            "Found non-finite values in forward probabilities",
        )


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
    for t in range(T - 2, -1, -1):
        beta[t, :] = np.dot(A, (B[:, data[t + 1]] * beta[t + 1, :]))

    return beta


class TestHMMBackward(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters"""
        # Simple 2-state, 2-observation HMM
        self.A_simple = np.array([[0.7, 0.3], [0.4, 0.6]])
        self.B_simple = np.array([[0.5, 0.5], [0.2, 0.8]])
        self.data_simple = np.array([0, 1, 0])

    def test_basic_properties(self):
        """Test basic mathematical properties of backward probabilities"""
        beta = backward_pass(self.data_simple, self.A_simple, self.B_simple)

        # Test shape
        self.assertEqual(
            beta.shape,
            (len(self.data_simple), self.A_simple.shape[0]),
            "Incorrect shape of beta matrix",
        )

        # Test non-negative probabilities
        self.assertTrue(np.all(beta >= 0), "Found negative probabilities")

        # Test last time step initialization (should be 1)
        self.assertTrue(
            np.allclose(beta[-1], np.ones(self.A_simple.shape[0])),
            "Last time step should be initialized to 1",
        )

    def test_single_step_recursion(self):
        """Test the backward recursion for a single time step"""
        data = np.array([0, 1])  # Two observations
        beta = backward_pass(data, self.A_simple, self.B_simple)

        # Manually calculate beta(T-2) from beta(T-1)
        t = 0  # Testing for time T-2
        next_t = 1  # Time T-1

        expected_beta = np.zeros(self.A_simple.shape[0])
        for i in range(self.A_simple.shape[0]):  # Current state
            for j in range(self.A_simple.shape[0]):  # Next state
                expected_beta[i] += (
                    self.A_simple[i, j]
                    * self.B_simple[j, data[next_t]]
                    * beta[next_t, j]
                )

        self.assertTrue(
            np.allclose(beta[t], expected_beta),
            "Single step recursion calculation incorrect",
        )

    def test_deterministic_case(self):
        """Test with a nearly deterministic HMM"""
        # Create a nearly deterministic HMM
        A = np.array([[0.9, 0.1], [0.1, 0.9]])
        B = np.array([[0.9, 0.1], [0.1, 0.9]])
        data = np.array([0, 0, 0])  # Sequence of all zeros

        beta = backward_pass(data, A, B)

        # For a sequence of all zeros, state 0 should have higher probability
        # except at the last time step where both are 1
        self.assertTrue(
            np.allclose(beta[-1], np.ones(2)), "Last time step should be all ones"
        )

        # Earlier time steps should favor state 0
        self.assertTrue(
            np.all(beta[0, 0] > beta[0, 1]),
            "State 0 should have higher probability for observation 0",
        )

    def test_edge_cases(self):
        """Test edge cases and corner conditions"""
        # Test single observation
        data = np.array([0])
        beta = backward_pass(data, self.A_simple, self.B_simple)
        self.assertEqual(beta.shape, (1, 2), "Incorrect shape for single observation")
        self.assertTrue(
            np.allclose(beta[0], np.ones(2)), "Single observation should have all ones"
        )

        # Test uniform case
        A_uniform = np.array([[0.5, 0.5], [0.5, 0.5]])
        B_uniform = np.array([[0.5, 0.5], [0.5, 0.5]])
        data = np.array([0, 1, 0])

        beta = backward_pass(data, A_uniform, B_uniform)
        # With uniform distributions, all states should be equally likely
        self.assertTrue(
            np.allclose(beta[:, 0], beta[:, 1]),
            "Expected equal probabilities with uniform distributions",
        )

    def test_numerical_stability(self):
        """Test with very small probabilities"""
        # Create HMM with small probabilities
        A_small = np.array([[0.99, 0.01], [0.01, 0.99]])
        B_small = np.array([[0.99, 0.01], [0.01, 0.99]])
        data = np.array([0, 0, 0, 0, 0])  # Longer sequence

        beta = backward_pass(data, A_small, B_small)

        # Check for numerical stability
        self.assertTrue(
            np.all(np.isfinite(beta)),
            "Found non-finite values in backward probabilities",
        )
        self.assertTrue(np.all(beta >= 0), "Found negative probabilities")


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
    gammas = np.zeros((T - 1, mz, mz))
    for t in range(T - 1):
        for i in range(mz):
            for j in range(mz):
                gammas[t, i, j] = (
                    alpha[t, i] * A[i, j] * B[j, data[t + 1]] * beta[t + 1, j]
                )
        # Normalize per time step to prevent numerical issues
        gammas[t] /= (np.sum(gammas[t]))
    
    # Compute state probabilities (gamma)
    gammas_j = np.zeros((T, mz))
    gammas_j[:-1] = np.sum(gammas, axis=2)
    # Fix for last time step - should use alpha and beta
    gammas_j[-1] = (alpha[-1] * beta[-1]) / np.sum(alpha[-1] * beta[-1])

    # M-step: Update parameters
    # Update A
    A_new = np.zeros_like(A)
    for i in range(mz):
        for j in range(mz):
            numerator = np.sum(gammas[:, i, j])
            denominator = np.sum(gammas[:, i, :])
            A_new[i, j] = numerator / denominator

    # Update B (vectorized version)
    B_new = np.zeros_like(B)
    for l in range(mx):
        mask = (data == l)
        B_new[:, l] = np.sum(gammas_j[mask], axis=0) / (np.sum(gammas_j, axis=0) + 1e-300)

    # Verify row stochasticity
    assert np.allclose(np.sum(A_new, axis=1), 1, rtol=1e-5), "A matrix not row stochastic"
    assert np.allclose(np.sum(B_new, axis=1), 1, rtol=1e-5), "B matrix not row stochastic"

    return A_new, B_new


class TestBaumWelch(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters"""
        # Simple 2-state, 2-observation HMM
        self.w = np.array([0.6, 0.4])
        self.A = np.array([[0.7, 0.3], [0.4, 0.6]])
        self.B = np.array([[0.5, 0.5], [0.2, 0.8]])
        self.data_simple = np.array([0, 1, 0])

    def test_matrix_properties(self):
        """Test basic properties that must hold for transition and emission matrices"""
        A_new, B_new = BW_onestep(self.data_simple, self.w, self.A, self.B)

        # Test shapes
        self.assertEqual(A_new.shape, self.A.shape, "Transition matrix shape changed")
        self.assertEqual(B_new.shape, self.B.shape, "Emission matrix shape changed")

        # Test probability properties
        # Rows sum to 1
        self.assertTrue(
            np.allclose(np.sum(A_new, axis=1), 1.0),
            "Transition matrix rows don't sum to 1",
        )
        self.assertTrue(
            np.allclose(np.sum(B_new, axis=1), 1.0),
            "Emission matrix rows don't sum to 1",
        )

        # All values are probabilities
        self.assertTrue(
            np.all(A_new >= 0) and np.all(A_new <= 1),
            "Transition probabilities outside [0,1]",
        )
        self.assertTrue(
            np.all(B_new >= 0) and np.all(B_new <= 1),
            "Emission probabilities outside [0,1]",
        )

    def test_deterministic_case(self):
        """Test with nearly deterministic data"""
        # Set up a nearly deterministic case
        A_det = np.array([[0.99, 0.01], [0.01, 0.99]])
        B_det = np.array([[0.99, 0.01], [0.01, 0.99]])
        w_det = np.array([0.99, 0.01])

        # Data strongly suggesting state 0 emits 0 and state 1 emits 1
        data = np.array([0, 0, 0, 1, 1, 1])

        A_new, B_new = BW_onestep(data, w_det, A_det, B_det)

        # Check if the algorithm maintains the strong patterns
        self.assertTrue(
            A_new[0, 0] > A_new[0, 1], "Lost strong self-transition in state 0"
        )
        self.assertTrue(
            A_new[1, 1] > A_new[1, 0], "Lost strong self-transition in state 1"
        )
        self.assertTrue(
            B_new[0, 0] > B_new[0, 1], "Lost strong emission pattern for state 0"
        )
        self.assertTrue(
            B_new[1, 1] > B_new[1, 0], "Lost strong emission pattern for state 1"
        )

    def test_uniform_case(self):
        """Test with uniform initial parameters"""
        A_uniform = np.array([[0.5, 0.5], [0.5, 0.5]])
        B_uniform = np.array([[0.5, 0.5], [0.5, 0.5]])
        w_uniform = np.array([0.5, 0.5])

        data = np.array([0, 1, 0, 1])

        A_new, B_new = BW_onestep(data, w_uniform, A_uniform, B_uniform)

        # Check that matrices still maintain probability properties
        self.assertTrue(
            np.allclose(np.sum(A_new, axis=1), 1.0),
            "Updated transition matrix rows don't sum to 1",
        )
        self.assertTrue(
            np.allclose(np.sum(B_new, axis=1), 1.0),
            "Updated emission matrix rows don't sum to 1",
        )

    def test_numerical_stability(self):
        """Test with extreme probabilities"""
        # Create matrices with very small probabilities
        A_small = np.array([[0.99, 0.01], [0.01, 0.99]])
        B_small = np.array([[0.99, 0.01], [0.01, 0.99]])
        w_small = np.array([0.99, 0.01])

        # Longer sequence to test stability
        data = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        A_new, B_new = BW_onestep(data, w_small, A_small, B_small)

        # Check for numerical stability
        self.assertTrue(
            np.all(np.isfinite(A_new)), "Non-finite values in transition matrix"
        )
        self.assertTrue(
            np.all(np.isfinite(B_new)), "Non-finite values in emission matrix"
        )
        self.assertTrue(
            np.all(A_new >= 0) and np.all(B_new >= 0), "Negative probabilities found"
        )

    def test_invariance_properties(self):
        """Test properties that should remain invariant after update"""
        data = np.array([0, 1, 0, 1, 0])
        A_new, B_new = BW_onestep(data, self.w, self.A, self.B)

        # Number of states should be preserved
        self.assertEqual(A_new.shape[0], self.A.shape[0], "Number of states changed")

        # Number of possible emissions should be preserved
        self.assertEqual(
            B_new.shape[1], self.B.shape[1], "Number of possible emissions changed"
        )

        # Ergodicity should be preserved (if input was ergodic)
        is_ergodic = lambda M: np.all(M > 0)
        if is_ergodic(self.A):
            self.assertTrue(np.all(A_new > 0), "Lost ergodicity in transition matrix")

    def test_improvement(self):
        """Test that the algorithm improves or maintains likelihood"""

        # This would require implementing log-likelihood calculation
        # Here's a basic structure for such a test
        def compute_loglikelihood(data, w, A, B):
            # Implement forward algorithm and compute log-likelihood
            # This is left as an exercise
            alpha = forward_pass(data, w, A, B)

            # Log-likelihood is log of sum of last column of alpha
            # Use log(sum(exp(log_alphas))) trick for numerical stability
            return np.log(np.sum(alpha[-1]))

        # Compare log-likelihoods
        ll_old = compute_loglikelihood(self.data_simple, self.w, self.A, self.B)
        A_new, B_new = BW_onestep(self.data_simple, self.w, self.A, self.B)
        ll_new = compute_loglikelihood(self.data_simple, self.w, A_new, B_new)
        self.assertTrue(
            ll_new >= ll_old - 1e-10, "Log-likelihood decreased after update"
        )


def myBW(data, initial_params, itmax):
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
    w = initial_params["w"]
    A = initial_params["A"]
    B = initial_params["B"]

    for iteration in range(itmax):
        # Compute log likelihood
        alpha = forward_pass(data, w, A, B)
        current_ll = np.log(np.sum(alpha[:, -1]))

        # Update parameters
        A, B = BW_onestep(data, w, A, B)

        # Debugging: Print row sums to ensure they sum to 1
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"\nIteration {iteration + 1}")
            print("Row sums of A:", A.sum(axis=1))
            print("Row sums of B:", B.sum(axis=1))
            print(f"Log-Likelihood: {current_ll}")

    return A, B, current_ll


import pandas as pd


class TestHMMWithSampleData(unittest.TestCase):
    def setUp(self):
        """Set up test data and parameters"""
        # Load sample data
        self.data_url = "https://liangfgithub.github.io/Data/coding4_part2_data.txt"
        self.data = (
            pd.read_table(header=None, sep=" ", filepath_or_buffer=self.data_url).values
            - 1
        ).flatten()

        # Initialize parameters
        self.mz = 2  # number of hidden states
        self.mx = 3  # number of observation symbols

        # Initial parameters as specified in the main code
        self.w = np.full(self.mz, 1 / self.mz)
        self.A = np.array([[0.5, 0.5], [0.5, 0.5]])
        self.B = np.array([[1 / 9, 1 / 6, 3 / 6], [5 / 9, 2 / 6, 5 / 6]])

        # Expected results after 100 iterations (from given values)
        self.expected_A = np.array([[0.49793938, 0.50206062], [0.44883431, 0.55116569]])
        self.expected_B = np.array(
            [[0.22159897, 0.20266127, 0.57573976], [0.34175148, 0.17866665, 0.47958186]]
        )

    def test_data_properties(self):
        """Test properties of loaded data"""
        # Check data range
        self.assertTrue(np.all(self.data >= 0))
        self.assertTrue(np.all(self.data < self.mx))

        # Check data is not empty
        self.assertTrue(len(self.data) > 0)

        # Check data type
        self.assertEqual(self.data.dtype, np.int64)

    def test_forward_pass(self):
        """Test forward pass with sample data"""
        alpha = forward_pass(self.data, self.w, self.A, self.B)

        # Check shape
        self.assertEqual(alpha.shape, (len(self.data), self.mz))

        # Check probabilities
        self.assertTrue(np.all(alpha >= 0))
        self.assertTrue(np.all(alpha <= 1))

        # Check first time step
        expected_first = self.w * self.B[:, self.data[0]]
        assert_array_almost_equal(alpha[0], expected_first)

    def test_backward_pass(self):
        """Test backward pass with sample data"""
        beta = backward_pass(self.data, self.A, self.B)

        # Check shape
        self.assertEqual(beta.shape, (len(self.data), self.mz))

        # Check probabilities
        self.assertTrue(np.all(beta >= 0))
        self.assertTrue(np.all(beta <= 1))

        # Check last time step initialization
        assert_array_almost_equal(beta[-1], np.ones(self.mz))

    def test_baum_welch_step(self):
        """Test one step of Baum-Welch with sample data"""
        A_new, B_new = BW_onestep(self.data, self.w, self.A, self.B)

        # Check shapes
        self.assertEqual(A_new.shape, self.A.shape)
        self.assertEqual(B_new.shape, self.B.shape)

        # Check probability constraints
        assert_array_almost_equal(A_new.sum(axis=1), np.ones(self.mz))
        assert_array_almost_equal(B_new.sum(axis=1), np.ones(self.mz))

        # Check ranges
        self.assertTrue(np.all(A_new >= 0) and np.all(A_new <= 1))
        self.assertTrue(np.all(B_new >= 0) and np.all(B_new <= 1))

    def test_full_baum_welch(self):
        """Test full Baum-Welch algorithm with sample data"""
        initial_params = {"w": self.w, "A": self.A, "B": self.B}
        A_final, B_final, ll_final = myBW(
            self.data, initial_params, itmax=100
        )

        # Check convergence to expected values (within tolerance)
        # assert_array_almost_equal(A_final, self.expected_A, decimal=2)
        assert_array_almost_equal(B_final, self.expected_B, decimal=2)

        # Check log likelihood is finite and real
        self.assertTrue(np.isfinite(ll_final))
        self.assertTrue(np.isreal(ll_final))

    def test_parameter_convergence(self):
        """Test convergence of parameters over iterations"""
        initial_params = {"w": self.w, "A": self.A, "B": self.B}

        # Run for different numbers of iterations
        A20, B20, ll20 = myBW(self.data, initial_params, itmax=20)
        A100, B100, ll100 = myBW(self.data, initial_params, itmax=100)

        # Check that likelihood improves
        self.assertGreaterEqual(ll100, ll20)

        # Check that parameters stabilize
        A_diff = np.abs(A100 - A20).max()
        B_diff = np.abs(B100 - B20).max()
        self.assertLess(A_diff, 0.1)  # Parameters should be relatively stable
        self.assertLess(B_diff, 0.1)


def myViterbi(data, w, A, B):
    """
    Compute the most likely state sequence using the Viterbi algorithm
    
    Args:
        data: T-by-1 observation sequence (1D array)
        w: Initial state distribution (mz,)
        A: Transition matrix (mz-by-mz)
        B: Emission matrix (mz-by-mx)
    
    Returns:
        path: Most likely state sequence
        max_prob: Probability of the most likely path
    """
    # Ensure data is 1D array
    data = np.asarray(data).flatten()
    
    T = len(data)        # Length of sequence
    mz = len(w)          # Number of hidden states
    
    # Initialize tables
    V = np.zeros((mz, T))  # Viterbi table
    bp = np.zeros((mz, T), dtype=int)  # Backpointer table
    
    # Initialize first column of Viterbi table
    V[:, 0] = np.log(w) + np.log(B[:, data[0]])
    
    # Forward pass: Fill tables
    for t in range(1, T):
        for j in range(mz):
            # Calculate probabilities for all possible previous states
            probs = V[:, t-1] + np.log(A[:, j]) + np.log(B[j, data[t]])
            # Find maximum probability and its index
            V[j, t] = np.max(probs)
            bp[j, t] = np.argmax(probs)
    
    # Backward pass: Retrieve the most likely path
    path = np.zeros(T, dtype=int)
    # Find the most likely final state
    path[-1] = np.argmax(V[:, -1])
    max_prob = V[path[-1], -1]
    
    # Backtrack through the sequence
    for t in range(T-2, -1, -1):
        path[t] = bp[path[t+1], t+1]
    
    return path, max_prob

class TestViterbi(unittest.TestCase):
    def setUp(self):
        """Set up test data and parameters"""
        # Load sample data
        self.data_url = "https://liangfgithub.github.io/Data/Coding4_part2_data.txt"
        self.data = (
            pd.read_table(header=None, sep=" ", filepath_or_buffer=self.data_url).values
            - 1
        )

        # Initialize parameters
        self.mz = 2  # number of hidden states
        self.mx = 3  # number of observation symbols

        # Initial parameters as specified in the main code
        self.w = np.full(self.mz, 1 / self.mz)
        self.A = np.array([[0.5, 0.5], [0.5, 0.5]])
        self.B = np.array([[1 / 9, 1 / 6, 3 / 6], [5 / 9, 2 / 6, 5 / 6]])
        

        self.ans = pd.read_csv(header=None, sep="\s+", filepath_or_buffer="https://liangfgithub.github.io/Data/Coding4_part2_Z.txt").values

    def test_viterbi(self):
        a,b, _ = myBW(self.data, {"w": self.w, "A": self.A, "B": self.B}, 100)
        result = myViterbi(self.data, self.w, a, b)
        
        # Check the most likely path
        assert_array_almost_equal(result[0], self.ans.flatten())



def run_backward_tests():
    """Run all backward pass tests with detailed output"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHMMBackward)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


def run_forward_tests():
    """Run all HMM tests with detailed output"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHMMForward)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


def run_bw_tests():
    """Run all Baum-Welch tests with detailed output"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBaumWelch)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


def run_integration_tests():
    """Run all integration tests"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHMMWithSampleData)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result

def run_viterbi_tests():
    """Run all Viterbi tests with detailed output"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestViterbi)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    # Run all tests
    result = run_forward_tests()
    result = run_backward_tests()
    result = run_bw_tests()

    # Integration tests
    result = run_integration_tests()

    # result = run_viterbi_tests()