# MemFormer

<img width="763" alt="image" src="https://github.com/user-attachments/assets/ec3902b0-2bfd-4021-af5d-968f3e311d57" />

MemFormer is a memory-augmented generative model designed for time series generation, integrating a Transformer architecture with an adapter-memory module to efficiently stores and retrieves temporal patterns. The model introduces a reinforcement learning-based noise sampling mechanism that enhances the diversity of the generated data. Overall, MemFormer learns to generate more varied synthetic sequences, improving the effectiveness of data augmentation, especially in data-scarce environments.

## Key Components

### MemTCN (Memory-Augmented Temporal Convolution Network)
Combines convolution layer with an adapter-memory module.

<img width="712" alt="image" src="https://github.com/user-attachments/assets/414c390f-0e9b-454f-ba6e-3092eced8c4a" />

**Chunking Operation**: To improve the model’s scalability and efficiency, we implement the chunking operation introduced [here](https://github.com/salesforce/fsnet?tab=readme-ov-file):
- Flatten the gradient EMA into a vector.
- Split the gradient vector into $d$ chunks.
- Map each chunk to a hidden representation.
- Map each hidden representation to a coordinate of the target adaptation parameter.

This reduces both training time and computational cost.

### RL-Based Sampler
The sampler treats the MemFormer as the environment and samples proper input to maximize diversity among generated data. This approach enhances the overall performance of MemFormer, especially in situations where training data is limited.

## Experimental Configuration

Below are the experimental configurations used to train and evaluate MemFormer:

- **Iterations**: 50,000 for MemFormer and 10,000 for Sampler
- **Learning Rate**: 1e-4 for MemFormer and 1e-3 for Sampler
- **Batch Size**: 128
- **Optimizer**: Adam
- **Chunk Size**: 32
- **Noise Sampling Method**: Mixture of Gaussian
