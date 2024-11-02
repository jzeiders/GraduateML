import numpy as np
from numpy.linalg import inv, det
from scipy.stats import multivariate_normal

def Estep(data, prob, mean, Sigma):
    """
    E-step: Calculate responsibilities
    
    Parameters:
    -----------
    data: ndarray, shape (n, p)
        The input data matrix
    prob: ndarray, shape (G,)
        Mixing proportions
    mean: ndarray, shape (p, G)
        Mean vectors for each component
    Sigma: ndarray, shape (p, p)
        Shared covariance matrix
    
    Returns:
    --------
    responsibilities: ndarray, shape (n, G)
        Matrix of posterior probabilities P(Z_i=k|x_i)
    """
    n = data.shape[0]
    G = prob.shape[0]
    
    # Initialize matrix to store densities
    densities = np.zeros((n, G))
    
    # Compute gaussian densities for each component
    for k in range(G):
        densities[:, k] = multivariate_normal.pdf(data, mean[:, k], Sigma)
    
    # Multiply by mixing proportions
    weighted_densities = densities * prob[np.newaxis, :]
    
    # Normalize to get responsibilities
    responsibilities = weighted_densities / np.sum(weighted_densities, axis=1)[:, np.newaxis]
    
    return responsibilities

def Mstep(data, responsibilities):
    """
    M-step: Update parameters
    
    Parameters:
    -----------
    data: ndarray, shape (n, p)
        The input data matrix
    responsibilities: ndarray, shape (n, G)
        Matrix of posterior probabilities
        
    Returns:
    --------
    prob: ndarray, shape (G,)
        Updated mixing proportions
    mean: ndarray, shape (p, G)
        Updated mean vectors
    Sigma: ndarray, shape (p, p)
        Updated shared covariance matrix
    """
    n, p = data.shape
    G = responsibilities.shape[1]
    
    # Update mixing proportions
    prob = np.mean(responsibilities, axis=0)
    
    # Update means
    mean = np.zeros((p, G))
    for k in range(G):
        mean[:, k] = np.sum(responsibilities[:, k:k+1] * data, axis=0) / np.sum(responsibilities[:, k])
    
    # Update shared covariance matrix
    Sigma = np.zeros((p, p))
    for k in range(G):
        diff = data - mean[:, k]  # Shape: (n, p)
        weighted_diff = responsibilities[:, k].reshape(n, 1) * diff  # Shape: (n, p)
        Sigma += weighted_diff.T @ diff  # Shape: (p, p)
    Sigma /= n
    
    return prob, mean, Sigma

def loglik(data, prob, mean, Sigma):
    """
    Compute log-likelihood
    
    Parameters:
    -----------
    data: ndarray, shape (n, p)
        The input data matrix
    prob: ndarray, shape (G,)
        Mixing proportions
    mean: ndarray, shape (p, G)
        Mean vectors
    Sigma: ndarray, shape (p, p)
        Shared covariance matrix
        
    Returns:
    --------
    ll: float
        Log-likelihood value
    """
    n = data.shape[0]
    G = prob.shape[0]
    
    # Initialize array for component densities
    densities = np.zeros((n, G))
    
    # Compute densities for each component
    for k in range(G):
        densities[:, k] = prob[k] * multivariate_normal.pdf(data, mean[:, k], Sigma)
    
    # Sum over components and take log
    ll = np.sum(np.log(np.sum(densities, axis=1)))
    
    return ll

def myEM(data, G, initial_params, itmax):
    """
    Main EM algorithm function
    
    Parameters:
    -----------
    data: ndarray, shape (n, p)
        The input data matrix
    G: int
        Number of components
    initial_params: dict
        Dictionary containing initial parameters:
        'prob': mixing proportions
        'mean': mean vectors
        'Sigma': covariance matrix
    itmax: int
        Maximum number of iterations
        
    Returns:
    --------
    prob: ndarray, shape (G,)
        Final mixing proportions
    mean: ndarray, shape (p, G)
        Final mean vectors
    Sigma: ndarray, shape (p, p)
        Final shared covariance matrix
    ll: float
        Final log-likelihood value
    """
    # Initialize parameters
    prob = initial_params['prob']
    mean = initial_params['mean']
    Sigma = initial_params['Sigma']
    
    for _ in range(itmax):
        # E-step
        responsibilities = Estep(data, prob, mean, Sigma)
        
        # M-step
        prob, mean, Sigma = Mstep(data, responsibilities)
    
    # Compute final log-likelihood
    ll = loglik(data, prob, mean, Sigma)
    
    return prob, mean, Sigma, ll

# Test function for G=2
def test_G2(data):
    """Test EM algorithm with G=2 components"""
    n, p = data.shape
    
    # Set initial parameters as specified
    p1 = 10/n
    prob = np.array([p1, 1-p1])
    
    mean = np.zeros((p, 2))
    mean[:, 0] = np.mean(data[:10], axis=0)  # mean of first 10 samples
    mean[:, 1] = np.mean(data[10:], axis=0)  # mean of remaining samples
    
    # Calculate initial Sigma
    diff1 = data[:10] - mean[:, 0]
    diff2 = data[10:] - mean[:, 1]
    Sigma = (diff1.T @ diff1 + diff2.T @ diff2) / n
    
    initial_params = {
        'prob': prob,
        'mean': mean,
        'Sigma': Sigma
    }
    
    # Run EM algorithm
    final_prob, final_mean, final_Sigma, final_ll = myEM(data, 2, initial_params, 20)
    
    return final_prob, final_mean, final_Sigma, final_ll

# Test function for G=3
def test_G3(data):
    """Test EM algorithm with G=3 components"""
    n, p = data.shape
    
    # Set initial parameters
    p1, p2 = 10/n, 20/n
    prob = np.array([p1, p2, 1-p1-p2])
    
    mean = np.zeros((p, 3))
    mean[:, 0] = np.mean(data[:10], axis=0)      # mean of first 10 samples
    mean[:, 1] = np.mean(data[10:30], axis=0)    # mean of next 20 samples
    mean[:, 2] = np.mean(data[30:], axis=0)      # mean of remaining samples
    
    # Calculate initial Sigma
    diff1 = data[:10] - mean[:, 0]
    diff2 = data[10:30] - mean[:, 1]
    diff3 = data[30:] - mean[:, 2]
    Sigma = (diff1.T @ diff1 + diff2.T @ diff2 + diff3.T @ diff3) / n
    
    initial_params = {
        'prob': prob,
        'mean': mean,
        'Sigma': Sigma
    }
    
    # Run EM algorithm
    final_prob, final_mean, final_Sigma, final_ll = myEM(data, 3, initial_params, 20)
    
    return final_prob, final_mean, final_Sigma, final_ll

# Main testing code
if __name__ == "__main__":
    # Load faithful dataset
    file_path = '/Users/jzeiders/Documents/Code/Learnings/GraduateML/src/Coding4/faithful.dat'
    data = np.loadtxt(file_path, skiprows=1,dtype=(float, float), usecols=(1,2))
    print(data.shape) 
    print(data[:5])

    
    # Test G=2
    print("\nTesting with G=2:")
    prob2, mean2, Sigma2, ll2 = test_G2(data)
    
    print("\nprob")
    print(prob2)
    print("\nmean")
    print(mean2)
    print("\nSigma")
    print(Sigma2)
    print("\nloglik")
    print(ll2)
    
    # Test G=3
    print("\nTesting with G=3:")
    prob3, mean3, Sigma3, ll3 = test_G3(data)
    
    print("\nprob")
    print(prob3)
    print("\nmean")
    print(mean3)
    print("\nSigma")
    print(Sigma3)
    print("\nloglik")
    print(ll3)