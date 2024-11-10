import unittest
import numpy as np



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


class TestHMMForward(unittest.TestCase):
    def setUp(self):
        """Set up some common HMM parameters that will be used across tests"""
        # Simple 2-state, 2-observation HMM
        self.w_simple = np.array([0.6, 0.4])
        self.A_simple = np.array([[0.7, 0.3],
                                [0.4, 0.6]])
        self.B_simple = np.array([[0.5, 0.5],
                                [0.2, 0.8]])
        self.data_simple = np.array([0, 1, 0])

    def test_basic_properties(self):
        """Test basic mathematical properties of forward probabilities"""
        alpha = forward_pass(self.data_simple, self.w_simple, 
                           self.A_simple, self.B_simple)
        
        # Test non-negative probabilities
        self.assertTrue(np.all(alpha >= 0), 
                       "Found negative probabilities")
        
        # Test sum of probabilities <= 1
        self.assertTrue(np.all(np.sum(alpha, axis=1) <= 1 + 1e-10), 
                       "Probabilities sum > 1")
        
        # Test first time step calculation
        expected_first = self.w_simple * self.B_simple[:, self.data_simple[0]]
        self.assertTrue(np.allclose(alpha[0], expected_first),
                       "First step calculation incorrect")

    def test_deterministic_sequence(self):
        """Test with a nearly deterministic HMM"""
        w = np.array([1.0, 0.0])  # Start in state 0
        A = np.array([[0.9, 0.1],  # Mostly stay in same state
                     [0.1, 0.9]])
        B = np.array([[0.9, 0.1],  # State 0 mostly emits 0
                     [0.1, 0.9]])  # State 1 mostly emits 1
        data = np.array([0, 0, 0])
        
        alpha = forward_pass(data, w, A, B)
        self.assertTrue(np.all(alpha[:, 0] > alpha[:, 1]),
                       "Expected state 0 to have higher probability")

    def test_edge_cases(self):
        """Test edge cases and corner conditions"""
        # Test single observation
        w = np.array([0.5, 0.5])
        A = np.array([[0.5, 0.5],
                     [0.5, 0.5]])
        B = np.array([[0.5, 0.5],
                     [0.5, 0.5]])
        data = np.array([0])
        
        alpha = forward_pass(data, w, A, B)
        self.assertEqual(alpha.shape, (1, 2),
                        "Incorrect shape for single observation")
        
        # Test uniform distributions
        data = np.array([0, 1, 0])
        alpha = forward_pass(data, w, A, B)
        self.assertTrue(np.allclose(alpha[:, 0], alpha[:, 1]),
                       "Expected equal probabilities with uniform distributions")

    def test_known_sequence(self):
        """Test with manually calculated expected values"""
        w = np.array([0.8, 0.2])
        A = np.array([[0.6, 0.4],
                     [0.3, 0.7]])
        B = np.array([[0.7, 0.3],
                     [0.1, 0.9]])
        data = np.array([0, 1])
        
        alpha = forward_pass(data, w, A, B)
        
        # Check first time step
        expected_t0 = w * B[:, data[0]]
        self.assertTrue(np.allclose(alpha[0], expected_t0),
                       "First time step incorrect")
        
        # Check second time step
        expected_t1 = B[:, data[1]] * (A.T @ alpha[0])
        self.assertTrue(np.allclose(alpha[1], expected_t1),
                       "Second time step incorrect")

    def test_numerical_stability(self):
        """Test with very small probabilities"""
        w = np.array([0.99, 0.01])
        A = np.array([[0.99, 0.01],
                     [0.01, 0.99]])
        B = np.array([[0.99, 0.01],
                     [0.01, 0.99]])
        data = np.array([0, 0, 0, 0, 0])  # Long sequence to test stability
        
        alpha = forward_pass(data, w, A, B)
        self.assertTrue(np.all(np.isfinite(alpha)),
                       "Found non-finite values in forward probabilities")

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

class TestHMMBackward(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters"""
        # Simple 2-state, 2-observation HMM
        self.A_simple = np.array([[0.7, 0.3],
                                [0.4, 0.6]])
        self.B_simple = np.array([[0.5, 0.5],
                                [0.2, 0.8]])
        self.data_simple = np.array([0, 1, 0])

    def test_basic_properties(self):
        """Test basic mathematical properties of backward probabilities"""
        beta = backward_pass(self.data_simple, self.A_simple, self.B_simple)
        
        # Test shape
        self.assertEqual(beta.shape, (len(self.data_simple), self.A_simple.shape[0]),
                        "Incorrect shape of beta matrix")
        
        # Test non-negative probabilities
        self.assertTrue(np.all(beta >= 0),
                       "Found negative probabilities")
        
        # Test last time step initialization (should be 1)
        self.assertTrue(np.allclose(beta[-1], np.ones(self.A_simple.shape[0])),
                       "Last time step should be initialized to 1")

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
                expected_beta[i] += (self.A_simple[i, j] * 
                                   self.B_simple[j, data[next_t]] * 
                                   beta[next_t, j])
        
        self.assertTrue(np.allclose(beta[t], expected_beta),
                       "Single step recursion calculation incorrect")

    def test_deterministic_case(self):
        """Test with a nearly deterministic HMM"""
        # Create a nearly deterministic HMM
        A = np.array([[0.9, 0.1],
                     [0.1, 0.9]])
        B = np.array([[0.9, 0.1],
                     [0.1, 0.9]])
        data = np.array([0, 0, 0])  # Sequence of all zeros
        
        beta = backward_pass(data, A, B)
        
        # For a sequence of all zeros, state 0 should have higher probability
        # except at the last time step where both are 1
        self.assertTrue(np.allclose(beta[-1], np.ones(2)),
                       "Last time step should be all ones")
        
        # Earlier time steps should favor state 0
        self.assertTrue(np.all(beta[0, 0] > beta[0, 1]),
                       "State 0 should have higher probability for observation 0")

    def test_edge_cases(self):
        """Test edge cases and corner conditions"""
        # Test single observation
        data = np.array([0])
        beta = backward_pass(data, self.A_simple, self.B_simple)
        self.assertEqual(beta.shape, (1, 2),
                        "Incorrect shape for single observation")
        self.assertTrue(np.allclose(beta[0], np.ones(2)),
                       "Single observation should have all ones")
        
        # Test uniform case
        A_uniform = np.array([[0.5, 0.5],
                            [0.5, 0.5]])
        B_uniform = np.array([[0.5, 0.5],
                            [0.5, 0.5]])
        data = np.array([0, 1, 0])
        
        beta = backward_pass(data, A_uniform, B_uniform)
        # With uniform distributions, all states should be equally likely
        self.assertTrue(np.allclose(beta[:, 0], beta[:, 1]),
                       "Expected equal probabilities with uniform distributions")

    def test_numerical_stability(self):
        """Test with very small probabilities"""
        # Create HMM with small probabilities
        A_small = np.array([[0.99, 0.01],
                           [0.01, 0.99]])
        B_small = np.array([[0.99, 0.01],
                           [0.01, 0.99]])
        data = np.array([0, 0, 0, 0, 0])  # Longer sequence
        
        beta = backward_pass(data, A_small, B_small)
        
        # Check for numerical stability
        self.assertTrue(np.all(np.isfinite(beta)),
                       "Found non-finite values in backward probabilities")
        self.assertTrue(np.all(beta >= 0),
                       "Found negative probabilities")


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
    gammas = np.zeros((T-1, mz, mz))
    for t in range(T-1):
        for i in range(mz):
            for j in range(mz):
                gammas[t,i,j] = alpha[t,i] * A[i,j] * B[j,data[t+1]] * beta[t+1,j]
    
    
    
    # M-step: Update parameters
    # Update A
    A_new = np.zeros_like(A.shape)
    for i in range(mz):
        for j in range(mz):
            numerator = 0
            for t in range(T-1):
                numerator += gammas[t,i,j]
            denominator = 0
            for jprime in range(T-1):
                for t in range(T-1):
                    denominator += gammas[t,i,jprime]
            A_new = numerator / denominator

    
    # Update B
    B_new = np.zeros_like(B)
    for i in range(mz):
        for l in range(mx):
            numerator = 0
            denominator = 0
            for t in range(T-1):
                if data[t] == l:
                    for j in range(mz):
                        numerator += gammas[t,i,j]
                for j in range(mz):
                    numerator += gammas[t,i,j]

                
      
    
    return A_new, B_new

class TestBaumWelch(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters"""
        # Simple 2-state, 2-observation HMM
        self.w = np.array([0.6, 0.4])
        self.A = np.array([[0.7, 0.3],
                          [0.4, 0.6]])
        self.B = np.array([[0.5, 0.5],
                          [0.2, 0.8]])
        self.data_simple = np.array([0, 1, 0])

    def test_matrix_properties(self):
        """Test basic properties that must hold for transition and emission matrices"""
        A_new, B_new = BW_onestep(self.data_simple, self.w, self.A, self.B)
        
        # Test shapes
        self.assertEqual(A_new.shape, self.A.shape,
                        "Transition matrix shape changed")
        self.assertEqual(B_new.shape, self.B.shape,
                        "Emission matrix shape changed")
        
        # Test probability properties
        # Rows sum to 1
        self.assertTrue(np.allclose(np.sum(A_new, axis=1), 1.0),
                       "Transition matrix rows don't sum to 1")
        self.assertTrue(np.allclose(np.sum(B_new, axis=1), 1.0),
                       "Emission matrix rows don't sum to 1")
        
        # All values are probabilities
        self.assertTrue(np.all(A_new >= 0) and np.all(A_new <= 1),
                       "Transition probabilities outside [0,1]")
        self.assertTrue(np.all(B_new >= 0) and np.all(B_new <= 1),
                       "Emission probabilities outside [0,1]")

    def test_deterministic_case(self):
        """Test with nearly deterministic data"""
        # Set up a nearly deterministic case
        A_det = np.array([[0.99, 0.01],
                         [0.01, 0.99]])
        B_det = np.array([[0.99, 0.01],
                         [0.01, 0.99]])
        w_det = np.array([0.99, 0.01])
        
        # Data strongly suggesting state 0 emits 0 and state 1 emits 1
        data = np.array([0, 0, 0, 1, 1, 1])
        
        A_new, B_new = BW_onestep(data, w_det, A_det, B_det)
        
        # Check if the algorithm maintains the strong patterns
        self.assertTrue(A_new[0,0] > A_new[0,1],
                       "Lost strong self-transition in state 0")
        self.assertTrue(A_new[1,1] > A_new[1,0],
                       "Lost strong self-transition in state 1")
        self.assertTrue(B_new[0,0] > B_new[0,1],
                       "Lost strong emission pattern for state 0")
        self.assertTrue(B_new[1,1] > B_new[1,0],
                       "Lost strong emission pattern for state 1")

    def test_uniform_case(self):
        """Test with uniform initial parameters"""
        A_uniform = np.array([[0.5, 0.5],
                            [0.5, 0.5]])
        B_uniform = np.array([[0.5, 0.5],
                            [0.5, 0.5]])
        w_uniform = np.array([0.5, 0.5])
        
        data = np.array([0, 1, 0, 1])
        
        A_new, B_new = BW_onestep(data, w_uniform, A_uniform, B_uniform)
        
        # Check that matrices still maintain probability properties
        self.assertTrue(np.allclose(np.sum(A_new, axis=1), 1.0),
                       "Updated transition matrix rows don't sum to 1")
        self.assertTrue(np.allclose(np.sum(B_new, axis=1), 1.0),
                       "Updated emission matrix rows don't sum to 1")

    def test_numerical_stability(self):
        """Test with extreme probabilities"""
        # Create matrices with very small probabilities
        A_small = np.array([[0.99, 0.01],
                          [0.01, 0.99]])
        B_small = np.array([[0.99, 0.01],
                          [0.01, 0.99]])
        w_small = np.array([0.99, 0.01])
        
        # Longer sequence to test stability
        data = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        
        A_new, B_new = BW_onestep(data, w_small, A_small, B_small)
        
        # Check for numerical stability
        self.assertTrue(np.all(np.isfinite(A_new)),
                       "Non-finite values in transition matrix")
        self.assertTrue(np.all(np.isfinite(B_new)),
                       "Non-finite values in emission matrix")
        self.assertTrue(np.all(A_new >= 0) and np.all(B_new >= 0),
                       "Negative probabilities found")

    def test_invariance_properties(self):
        """Test properties that should remain invariant after update"""
        data = np.array([0, 1, 0, 1, 0])
        A_new, B_new = BW_onestep(data, self.w, self.A, self.B)
        
        # Number of states should be preserved
        self.assertEqual(A_new.shape[0], self.A.shape[0],
                        "Number of states changed")
        
        # Number of possible emissions should be preserved
        self.assertEqual(B_new.shape[1], self.B.shape[1],
                        "Number of possible emissions changed")
        
        # Ergodicity should be preserved (if input was ergodic)
        is_ergodic = lambda M: np.all(M > 0)
        if is_ergodic(self.A):
            self.assertTrue(np.all(A_new > 0),
                          "Lost ergodicity in transition matrix")

    def test_improvement(self):
        """Test that the algorithm improves or maintains likelihood"""
        # This would require implementing log-likelihood calculation
        # Here's a basic structure for such a test
        def compute_loglikelihood(data, w, A, B):
            # Implement forward algorithm and compute log-likelihood
            # This is left as an exercise
            pass
        
        # Compare log-likelihoods
        # ll_old = compute_loglikelihood(self.data_simple, self.w, self.A, self.B)
        # A_new, B_new = BW_onestep(self.data_simple, self.w, self.A, self.B)
        # ll_new = compute_loglikelihood(self.data_simple, self.w, A_new, B_new)
        # self.assertTrue(ll_new >= ll_old - 1e-10,
        #                "Log-likelihood decreased after update")

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

if __name__ == '__main__':
    # Run all tests
    result = run_forward_tests()
    result = run_backward_tests()
    result = run_bw_tests()
    
    # Print summary
    print("\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")