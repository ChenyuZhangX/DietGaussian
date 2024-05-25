
import numpy as np
from scipy import stats


def sample_t(radius, n_samples = 1):
    '''
    sample t from spherical points
    '''

    proposal = stats.multivariate_normal(mean = np.zeros(3), cov = np.eye(3))
    samples = proposal.rvs(n_samples)

    samples = samples / np.linalg.norm(samples, axis = 1)[:, None]
    samples = samples * radius
    
    return samples

if __name__ == "__main__":
    samples = sample_t(4, n_samples = 3)

