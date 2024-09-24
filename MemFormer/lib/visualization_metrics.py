import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualization(input_1, input_2, method='pca'):
    # Ensure we take a maximum of 1000 samples from each input
    num_samples_1 = min(1000, input_1.shape[0])
    num_samples_2 = min(1000, input_2.shape[0])
    
    # Randomly sample the inputs
    sampled_indices_1 = np.random.choice(input_1.shape[0], num_samples_1, replace=False)
    sampled_indices_2 = np.random.choice(input_2.shape[0], num_samples_2, replace=False)
    
    sampled_data_1 = input_1[sampled_indices_1]
    sampled_data_2 = input_2[sampled_indices_2]
    
    # Combine sampled data
    combined_data = np.concatenate((sampled_data_1, sampled_data_2), axis=0)
    
    # Reshape to 2D (num_samples, feature_dim)
    reshaped_data = combined_data.reshape(combined_data.shape[0], -1)
    
    # Apply PCA or t-SNE
    if method == 'pca':
        model = PCA(n_components=2)
    elif method == 'tsne':
        pass
    else:
        raise ValueError("Method must be either 'pca' or 'tsne'")
    
    transformed_data = model.fit_transform(reshaped_data)

    # Plot in 2D
    plt.figure(figsize=(8, 6))
    
    # Plot sampled points from input_1 in red
    plt.scatter(transformed_data[:num_samples_1, 0], transformed_data[:num_samples_1, 1], color='red', alpha=0.2,  label='Original')
    
    # Plot sampled points from input_2 in blue
    plt.scatter(transformed_data[num_samples_1:, 0], transformed_data[num_samples_1:, 1], color='blue', alpha=0.2, label='Synthetic')
    
    plt.title(f'{method.upper()} Visualization')
    plt.xlabel('x-PCA')
    plt.ylabel('y-PCA')
    plt.legend()
    plt.grid(False)
    plt.show()
