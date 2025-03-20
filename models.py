import torch
import torch.nn as nn
import torch.optim as optim
import einops


########################## NEURAL NETWORKS ##########################

# Implemented:
# - C Monotonic Gradient Network (C-MGN)
# - M Monotonic Gradient Network (M-MGN)
# - Multi-Layer Perceptron (MLP)


class CMGN(nn.Module):
    def __init__(self, hidden_dim, v_dim, x_dim, num_layers, activation):
        super(CMGN, self).__init__()
        self.W = nn.Linear(x_dim, hidden_dim, bias=False)
        self.V = nn.Linear(x_dim, v_dim, bias=False)
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_layers)])
        self.bias_L = nn.Parameter(torch.zeros(x_dim))
        self.hidden_dim = hidden_dim
        self.activation = activation

    def forward(self, x):
        Wx = self.W(x)
        z = Wx + self.biases[0]

        for i in range(1, len(self.biases)):
            z = self.activation(z)
            z = z + self.biases[i] + Wx  # Re-adding Wx matches the equation

        # Equation (6): C-MGN(x) = W^T σ_L(z_{L-1}) + V^T V x + b_L

        
        term1 = einops.einsum(self.activation(z), self.W.weight, 'batch hidden, hidden x_dim -> batch x_dim')

        term2 = einops.einsum(self.V(x), self.V.weight, 'batch v_dim, v_dim x_dim -> batch x_dim')
        out = term1 + term2 + self.bias_L             # Final output

        return out


class MMGN(nn.Module):
    def __init__(self, x_dim, v_dim, hidden_dim, num_layers, activation, scaling_function):
        super(MMGN, self).__init__()
        self.a = nn.Parameter(torch.zeros(x_dim))
        self.V = nn.Linear(x_dim, v_dim, bias=False)
        self.W_list = nn.ModuleList([nn.Linear(x_dim, hidden_dim, bias=False) for _ in range(num_layers)])      # W_k
        self.bias_list = nn.ParameterList([nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_layers)])   # b_k
        self.activation = activation                                                                            # \sigma_k
        self.scaling_function = scaling_function                                                                # s_k

    def forward(self, x):
        # V^T V x
        # print(x.shape)
        term1 = einops.einsum(self.V(x), self.V.weight, 'batch v_dim, v_dim x_dim -> batch x_dim')

        # Sum term
        sum_term = 0
        for k in range(len(self.W_list)):
            z_k = self.W_list[k](x) + self.bias_list[k]  # z_k = W_k x + b_k
            scale = self.scaling_function(z_k)  # s(z_k)
            # W_k^T \sigma_k(z_k)
            weighted_activation = einops.einsum(self.activation(z_k), self.W_list[k].weight, 'batch hidden, hidden x_dim -> batch x_dim')
            sum_term += scale * weighted_activation

        out = self.a + term1 + sum_term
        return out


class MLP(nn.Module):
    def __init__(self, x_dim, hidden_dim, num_layers, activation):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(x_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)] + [nn.Linear(hidden_dim, x_dim)])
        self.activation = activation

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x



########################## DISTRIBUTION DISTANCE METRICS ##########################


# Implemented:
# - Maximum Mean Discrepancy (MMD)
# - Dual Wasserstein Distance (Wasserstein GAN-style critic)
# - Gaussian Mixture Model (GMM) sampler with log-likelihood calculation

# More precisions:

# Maximum Mean Discrepancy (MMD)
# Name: MMDDistance
# Parameters: kernel_width (default 1.0)
# Usage: mmd = MMDDistance(kernel_width=1.0)
#         distance = mmd(x, y)

# Dual Wasserstein Distance
# Name: DualWassersteinDistance
# Parameters: input_dim, hidden_dim=64, critic_lr=0.0005, critic_iterations=5, clip_value=0.01
# Usage: wasserstein = DualWassersteinDistance(input_dim, hidden_dim=64, critic_lr=0.0005, critic_iterations=5, clip_value=0.01)
#        distance = wasserstein(x, y)
#        ... # do some training
#        wasserstein.update(x, y) # Update the critic network

# Gaussian Mixture Model (GMM) sampler
# Name: GMMSampler
# Parameters: means, covs, weights=None
# Usage: gmm = GMMSampler(means, covs, weights=None)
#        samples = gmm.sample(n_samples)
#        log_prob = gmm.log_likelihood(x)
#        neg_log_likelihood = gmm.negative_log_likelihood(x)
#        total_neg_log_likelihood = gmm.total_negative_log_likelihood(x)
#        avg_neg_log_likelihood = gmm.average_negative_log_likelihood(x)
#        ... # or fit a GMM to data
#        gmm = GMMSampler.fit_sklearn(x, n_components=None, covariance_type='full', random_state=None, **kwargs)
#        best_n, cv_scores = GMMSampler.cross_validate_n_components(x, n_range=range(1, 11), cv=5, covariance_type='full', random_state=None)


class GaussianKLDivergence:
    def __init__(self, target_mean, target_cov, device='cpu', bias=True):
        self.device = torch.device(device)
        self.target_mean = torch.as_tensor(target_mean, device=self.device)
        self.target_cov = torch.as_tensor(target_cov, device=self.device)
        self.bias = bias  # Controls covariance estimation bias

        # Precompute target distribution terms
        self.L_target = torch.linalg.cholesky(self.target_cov)
        self.logdet_target = 2 * torch.diag(self.L_target).log().sum()
        self.inv_target = torch.cholesky_inverse(self.L_target)

    def __call__(self, X, _):
        X = X.to(self.device)
        N, D = X.shape
        x_mean = X.mean(dim=0)
        x_cov = (X - x_mean).T @ (X - x_mean) / (N - int(not self.bias))

        # Regularize covariance
        x_cov += 1e-6 * torch.eye(D, device=self.device)

        # KL Computation (numerically stable)
        diff = self.target_mean - x_mean
        trace_term = torch.trace(self.inv_target @ x_cov)
        quad_term = diff @ self.inv_target @ diff
        logdet_x = 2 * torch.linalg.cholesky(x_cov).diag().log().sum()

        kl_div = 0.5 * (
            trace_term + quad_term - D 
            + self.logdet_target - logdet_x
        )
        
        return kl_div  # Returns scalar per batch



class MMDDistance:
    def __init__(self, kernel_width=1.0):
        self.kernel_width = kernel_width
    
    def gaussian_kernel(self, x, y):
        """Compute Gaussian kernel between all pairs of x and y."""
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        
        x = x.unsqueeze(1)  # (x_size, 1, dim)
        y = y.unsqueeze(0)  # (1, y_size, dim)
        
        kernel_input = (x - y).pow(2).sum(2) / self.kernel_width
        return torch.exp(-kernel_input)  # (x_size, y_size)
    
    def __call__(self, x, y):
        """Calculate MMD distance between samples x and y."""
        x_kernel = self.gaussian_kernel(x, x)
        y_kernel = self.gaussian_kernel(y, y)
        xy_kernel = self.gaussian_kernel(x, y)
        
        # Calculate MMD
        mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
        return mmd

class DualWassersteinDistance:
    def __init__(self, input_dim, hidden_dim=64, critic_lr=0.0005, 
                 critic_iterations=5, clip_value=0.01):
        """
        Initialize the Adversarial Wasserstein Distance calculator.
        
        This implements a Wasserstein GAN-style critic approach to estimate
        the Wasserstein distance between two distributions.
        
        Args:
            input_dim: Dimensionality of the input data
            hidden_dim: Size of hidden layers in the critic network
            critic_lr: Learning rate for the critic
            critic_iterations: Number of critic updates per distance calculation
            clip_value: Value to clip weights for Wasserstein constraint
        """
        self.input_dim = input_dim
        self.critic_iterations = critic_iterations
        self.clip_value = clip_value
        
        # Initialize critic network
        self.critic = MLP(input_dim, hidden_dim, num_layers=3, activation=nn.ReLU())
        
        # Initialize optimizer
        self.optimizer = optim.RMSprop(self.critic.parameters(), lr=critic_lr)
        
    def update(self, x, y):
        """
        Update the critic network to better approximate Wasserstein distance.
        
        Args:
            x: Samples from the first distribution (fake)
            y: Samples from the second distribution (real)
        """
        self.critic.train()

        # Ensure inputs don't require gradients for critic training
        x = x.detach()
        y = y.detach()
        
        # Train the critic multiple times
        for _ in range(self.critic_iterations):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute critic scores
            x_score = self.critic(x).mean()
            y_score = self.critic(y).mean()
            
            # Compute Wasserstein loss E[f(fake)] - E[f(real)] that we will minimize
            wasserstein_loss = x_score - y_score
            
            wasserstein_loss.backward()
            self.optimizer.step()
            
            # Clip weights to enforce Lipschitz constraint
            for param in self.critic.parameters():
                param.data.clamp_(-self.clip_value, self.clip_value)

        self.critic.eval()
    
    def __call__(self, x, y):
        """
        Calculate the approximate Wasserstein distance between x and y.
        
        Args:
            x: Samples from the first distribution (fake)
            y: Samples from the second distribution (real)
            
        Returns:
            Scalar tensor containing the estimated Wasserstein distance
        """
        x_score = self.critic(x).mean()
                    
        return -x_score.mean()
    

import torch
from torch.distributions import MultivariateNormal
import numpy as np
from sklearn.mixture import GaussianMixture

class GMMSampler:
    def __init__(self, means, covs, weights=None):
        """
        Initialize a GMM sampler with means, covariances, and weights
        
        Args:
            means: List of mean vectors for each component
            covs: List of covariance matrices for each component
            weights: Mixture weights (defaults to uniform)
        """
        self.n_components = len(means)
        self.means = [torch.tensor(m, dtype=torch.float32) for m in means]
        self.covs = [torch.tensor(c, dtype=torch.float32) for c in covs]
        
        if weights is None:
            self.weights = torch.ones(self.n_components) / self.n_components
        else:
            self.weights = torch.tensor(weights, dtype=torch.float32)
            self.weights = self.weights / self.weights.sum()  # Normalize weights
        
        self.distributions = [
            MultivariateNormal(self.means[i], self.covs[i])
            for i in range(self.n_components)
        ]
    
    def sample(self, n_samples):
        """Sample n_samples from the GMM"""
        # Choose components based on weights
        component_indices = torch.multinomial(
            self.weights, n_samples, replacement=True
        )
        
        # Sample from the selected components
        samples = torch.zeros(n_samples, len(self.means[0]))
        for i, comp_idx in enumerate(component_indices):
            samples[i] = self.distributions[comp_idx].sample()
        
        return samples
    
    def log_likelihood(self, x):
        """
        Compute the log-likelihood of the data given the GMM parameters
        
        Args:
            x (torch.Tensor): Input data of shape (batch_size, n_dimensions)
            
        Returns:
            torch.Tensor: The log-likelihood for each data point
        """
        batch_size = x.shape[0]
        
        # Compute log probability for each component
        log_probs = torch.zeros(batch_size, self.n_components)
        
        for k in range(self.n_components):
            # Get log probability from the component's distribution
            log_probs[:, k] = self.distributions[k].log_prob(x)
        
        # log p(x) = log(Σ_k π_k p_k(x))
        # Using the log-sum-exp trick for numerical stability
        log_weights = torch.log(self.weights)
        log_prob_mixture = torch.logsumexp(
            log_weights + log_probs, dim=1
        )
        
        return log_prob_mixture

    
    def np_likelihood(self, x):
        """
        Compute the sample-wise likelihood of the data given the GMM parameters.
        
        Args:
            x (np.ndarray): Input data of shape (batch_size, n_dimensions)
            
        Returns:
            np.ndarray: The likelihood for each data point
        """
        batch_size = x.shape[0]
        
        # Compute log probability for each component
        log_probs = np.zeros((batch_size, self.n_components))
        
        for k in range(self.n_components):
            # Iterate over the batch and compute log probability for each sample
            for i in range(batch_size):
                # Ensure that x[i] has shape (2,) and is passed correctly
                log_probs[i, k] = self.distributions[k].log_prob(
                    torch.tensor(x[i], dtype=torch.float32)  # Pass one sample at a time
                ).detach().cpu().numpy()
        
        # log π_k
        log_weights = np.log(self.weights.detach().cpu().numpy()).reshape(1, -1)  # Ensure correct broadcasting
        
        # Compute log p(x) = log(Σ_k π_k p_k(x)) using log-sum-exp
        log_prob_mixture = logsumexp(log_weights + log_probs, axis=1)

        return np.exp(log_prob_mixture)  # Convert log-likelihood to likelihood (optional)


    def negative_log_likelihood(self, x):
        """
        Compute the negative log-likelihood of the data given the GMM parameters
        
        Args:
            x (torch.Tensor): Input data of shape (batch_size, n_dimensions)
            
        Returns:
            torch.Tensor: The negative log-likelihood for each data point
        """
        return -self.log_likelihood(x)
    
    def total_negative_log_likelihood(self, x):
        """
        Compute the total negative log-likelihood across all data points
        
        Args:
            x (torch.Tensor): Input data of shape (batch_size, n_dimensions)
            
        Returns:
            torch.Tensor: The total negative log-likelihood (scalar)
        """
        return self.negative_log_likelihood(x).sum()
    
    def average_negative_log_likelihood(self, x):
        """
        Compute the average negative log-likelihood across all data points
        
        Args:
            x (torch.Tensor): Input data of shape (batch_size, n_dimensions)
            
        Returns:
            torch.Tensor: The average negative log-likelihood (scalar)
        """
        return self.negative_log_likelihood(x).mean()
    
    @classmethod
    def fit_sklearn(cls, x, n_components=None, covariance_type='full', random_state=None, **kwargs):
        """
        Fit a GMM to data using scikit-learn's GaussianMixture
        
        Args:
            x (torch.Tensor): Input data of shape (batch_size, n_dimensions)
            n_components (int): Number of mixture components
            covariance_type (str): Type of covariance parameters to use
                Must be one of: 'full', 'tied', 'diag', 'spherical'
            random_state (int): Random seed for reproducibility
            **kwargs: Additional arguments passed to sklearn.mixture.GaussianMixture
            
        Returns:
            GMMSampler: A new GMMSampler instance with parameters fitted to the data
        """
        # Convert to numpy if needed
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.array(x)
        
        # Determine number of components if not specified
        if n_components is None:
            n_components = min(len(x_np) // 10, 10)  # Heuristic
        
        # Create and fit sklearn GMM
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            **kwargs
        )
        gmm.fit(x_np)
        
        # Extract parameters
        means = [torch.tensor(mean, dtype=torch.float32) for mean in gmm.means_]
        
        # Handle different covariance types
        if covariance_type == 'full':
            covs = [torch.tensor(cov, dtype=torch.float32) for cov in gmm.covariances_]
        elif covariance_type == 'tied':
            covs = [torch.tensor(gmm.covariances_, dtype=torch.float32)] * n_components
        elif covariance_type == 'diag':
            covs = [torch.diag(torch.tensor(cov, dtype=torch.float32)) for cov in gmm.covariances_]
        elif covariance_type == 'spherical':
            covs = [torch.eye(x_np.shape[1], dtype=torch.float32) * cov for cov in gmm.covariances_]
        
        weights = torch.tensor(gmm.weights_, dtype=torch.float32)
        
        # Create new GMMSampler instance
        return cls(means, covs, weights)
    
    @classmethod
    def cross_validate_n_components(cls, x, n_range=range(1, 11), cv=5, covariance_type='full', random_state=None):
        """
        Perform cross-validation to find the optimal number of components
        
        Args:
            x (torch.Tensor): Input data of shape (batch_size, n_dimensions)
            n_range (iterable): Range of number of components to try
            cv (int): Number of cross-validation folds
            covariance_type (str): Type of covariance parameters to use
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (best_n_components, cv_scores) - the optimal number of components and all CV scores
        """
        # Convert to numpy if needed
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.array(x)
        
        n_samples = len(x_np)
        fold_size = n_samples // cv
        
        # Initialize scores
        mean_scores = []
        
        for n in n_range:
            cv_scores = []
            
            for i in range(cv):
                # Create train/test split
                test_indices = np.arange(i * fold_size, min((i + 1) * fold_size, n_samples))
                train_indices = np.setdiff1d(np.arange(n_samples), test_indices)
                
                x_train, x_test = x_np[train_indices], x_np[test_indices]
                
                # Fit GMM on training data
                gmm = GaussianMixture(
                    n_components=n,
                    covariance_type=covariance_type,
                    random_state=random_state
                )
                gmm.fit(x_train)
                
                # Score on test data
                score = gmm.score(x_test)
                cv_scores.append(score)
            
            mean_scores.append(np.mean(cv_scores))
            print(f"{n} components: avg log-likelihood = {np.mean(cv_scores):.4f}")
        
        best_n = n_range[np.argmax(mean_scores)]
        return best_n, mean_scores