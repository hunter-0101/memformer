import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from data_loading import load_data, augment_data

from MemFormer.options import Options
from MemFormer.lib.data import load_data as load_data_memformer
from MemFormer.lib.memformer import MemFormer

import matplotlib.pyplot as plt
from MemFormer.lib.visualization_metrics import visualization

#%%
# Define the MoGSampler policy network
class MoGSampler(nn.Module):
    def __init__(self, z_dim=7, h_dim=24, seq_len=24, num_components=5, hidden_dim=64):
        super(MoGSampler, self).__init__()
        self.z_dim = z_dim
        self.seq_len = seq_len
        self.num_components = num_components

        # Layers to output means, variances, and mixing coefficients
        self.fc1 = nn.Linear(h_dim * seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, seq_len * num_components * z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, seq_len * num_components * z_dim)
        self.fc_weight = nn.Linear(hidden_dim, seq_len * num_components)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        # Forward pass through the network
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Reshape to get per-component parameters
        means = self.fc_mean(x).view(batch_size, self.seq_len, self.num_components, self.z_dim)
        logvars = self.fc_logvar(x).view(batch_size, self.seq_len, self.num_components, self.z_dim)
        weights = self.fc_weight(x).view(batch_size, self.seq_len, self.num_components)
        
        # Apply softmax to weights to ensure they sum to 1
        weights = torch.softmax(weights, dim=-1)
        
        # Apply exponential to logvars to get variances
        variances = torch.exp(logvars)
        
        return means, variances, weights

    def sample(self, means, variances, weights):
        batch_size, seq_len, num_components, z_dim = means.size()
        
        # Sample component indices based on the weights
        component_indices = torch.multinomial(weights.view(-1, num_components), 1).view(batch_size, seq_len)
        
        # Gather the corresponding means and variances
        selected_means = torch.gather(means, 2, component_indices.unsqueeze(-1).expand(-1, -1, -1, z_dim)).squeeze(2)
        selected_variances = torch.gather(variances, 2, component_indices.unsqueeze(-1).expand(-1, -1, -1, z_dim)).squeeze(2)
        
        # Sample from the selected Gaussian distributions
        std = torch.sqrt(selected_variances)
        z = torch.randn_like(selected_means) * std + selected_means
        
        return z

#%%
opt = Options().parse()
ori_data = load_data_memformer(opt)
memformer = MemFormer(opt, ori_data)
memformer.train()

memformer.evaluation()
print(memformer.metrics)

#%%
# Example state representation: Random noise or learned embedding
batch_size = 128
seq_len = 24
z_dim = 7
hidden_dim = 24
input_dim = z_dim  # Assuming the input_dim is same as z_dim for simplicity

# Initialize MoGSampler
sampler = MoGSampler(z_dim=z_dim, h_dim=hidden_dim, seq_len=seq_len, num_components=5)
optimizer = optim.Adam(sampler.parameters(), lr=0.001)

#%%
# Define the reward function based on pairwise distance
def compute_variety_loss(synthetic_data):
    # synthetic_data shape: (batch_size, seq_len, feature_dim)
    # We compute the pairwise distances across the batch
    synthetic_data = torch.tensor(synthetic_data, dtype=torch.float32)
    pairwise_dist = torch.cdist(synthetic_data.view(batch_size, -1), 
                                synthetic_data.view(batch_size, -1))
    # The reward is the sum of all pairwise distances
    reward = pairwise_dist.sum()
    return reward

#%%
# Training loop
num_episodes = 10000
#reward_history=[]
state = torch.randn(batch_size, seq_len, hidden_dim)
for episode in range(num_episodes):
    optimizer.zero_grad()
    
    # MoGSampler generates parameters for the MoG
    means, variances, weights = sampler(state)
    
    # Sample Z_t from the Mixture of Gaussians
    Z_t = torch.zeros(batch_size, seq_len, z_dim)
    log_probs = torch.zeros(batch_size, seq_len)

    for i in range(batch_size):
        for j in range(seq_len):
            # Create a categorical distribution for the mixture components
            component_distribution = torch.distributions.Categorical(weights[i, j])
            component_index = component_distribution.sample()  # Sample which component to use
            
            # Get the corresponding mean and variance
            selected_mean = means[i, j, component_index]
            selected_variance = variances[i, j, component_index]
            
            # Create a normal distribution for the selected component
            normal_distribution = torch.distributions.Normal(selected_mean, torch.sqrt(selected_variance))
            
            # Sample Z_t from the selected Gaussian
            Z_t[i, j] = normal_distribution.sample()
            
            # Calculate the log probability of the sampled Z_t
            log_prob = component_distribution.log_prob(component_index) + normal_distribution.log_prob(Z_t[i, j]).sum()
            log_probs[i, j] = log_prob

    # MemFormer generates synthetic data
    latent_representation, synthetic_data = memformer.frozen_generation(Z_t)
    state = latent_representation
    # Calculate reward (variety loss)
    reward = compute_variety_loss(synthetic_data)
    
    #reward_history.append(reward)
    # Policy gradient (REINFORCE)
    loss = -reward * log_probs.sum(dim=1).mean()  # Negate to maximize reward
    loss.backward()
    optimizer.step()

    if (episode + 1) % 20 == 0:
        print(f"Episode {episode + 1}, Reward: {reward.item()}")
        
        
'''
plt.figure()
plt.plot(range(1, 1001), reward_history)
plt.show()
'''