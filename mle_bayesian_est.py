import numpy as np
from scipy.optimize import minimize
from scipy.stats import wishart

file_data = None
omega2 = []

with open('Assignment3_dataset.txt', 'r') as file:
    file_data = file.readlines()
    for line in file_data:
        if not line.startswith('W'):
            temp_array = []
            for value in line.split('\t'):
                temp_array.append(value.strip())
            omega2.append(temp_array)

print(omega2)


# Function to calculate the negative log-likelihood for MLE
def neg_log_likelihood(params, data):
    mean = params[:3]
    diag_covariance = np.exp(params[3:])
    covariance = np.diag(diag_covariance)
    n = len(data)
    
    # Convert lists to NumPy arrays
    data = np.array(data, dtype=np.float64)
    mean = np.array(mean, dtype=np.float64)

    # Calculate the log-likelihood
    log_likelihood = -0.5 * (np.sum(np.log(np.linalg.det(covariance))) +
                            np.sum(np.linalg.solve(covariance, (data - mean).T).T * (data - mean)))

    
    # Return the negative log-likelihood
    return -log_likelihood / n

# Perform MLE optimization
initial_params = np.zeros(6)  # 3 for mean and 3 for log(diagonal covariance)
mle_result = minimize(neg_log_likelihood, initial_params, args=(omega2,), method='Nelder-Mead')

# Extract MLE estimates
mle_mean = mle_result.x[:3]
mle_diag_covariance = np.exp(mle_result.x[3:])
mle_covariance = np.diag(mle_diag_covariance)

print("MLE Mean:", mle_mean)
print("MLE Diagonal Covariance:", mle_diag_covariance)
print("MLE Covariance Matrix:")
print(mle_covariance)

# Bayesian estimation using a Wishart prior
n_samples = 1000
prior_scale = 1  # Adjust the scale parameter based on your prior beliefs

# Generate samples from the Wishart prior
prior_samples = wishart.rvs(df=n_samples, scale=prior_scale*np.eye(3))

# Convert omega2 to NumPy array with float64 data type
omega2 = np.array(omega2, dtype=np.float64)

# Bayesian estimation using the posterior mean
posterior_mean_covariance = np.linalg.inv(np.linalg.inv(prior_scale*np.eye(3)) + np.linalg.inv(np.cov(omega2, rowvar=False)))
posterior_mean_covariance = np.diag(np.diag(posterior_mean_covariance))
posterior_mean_mean = np.mean(omega2, axis=0)

print("\nBayesian Posterior Mean:")
print("Mean:", posterior_mean_mean)
print("Diagonal Covariance:", np.diag(posterior_mean_covariance))
print("Covariance Matrix:")
print(posterior_mean_covariance)
