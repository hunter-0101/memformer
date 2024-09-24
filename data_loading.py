import numpy as np

def load_data(file_path, num_rows=None, batch_size=24):
    # Load data from CSV file
    data = np.loadtxt(file_path, delimiter=',', skiprows=1, max_rows=num_rows)
    
    num_samples = data.shape[0] - batch_size + 1
    data_batches = [data[i:i + batch_size] for i in range(num_samples)]
    
    data_batches = np.array(data_batches)
    
    # Split into training and test sets
    train_size = int(len(data_batches) * 0.8)
    train_data = data_batches[:train_size]
    test_data = data_batches[train_size:]
    
    return train_data, test_data

def augment_data(original_data, generated_data):
    augmented_data = np.concatenate((original_data, generated_data), axis=0)
    np.random.shuffle(augmented_data)
    return augmented_data, original_data  # Use the original test data


#%%
def load_SA_data():
    X_sa = np.random.uniform(low=-10, high=10, size=5999)
    for t in range(1, 6000):
        if 1 <= t <= 1000:
            X_sa[t-1] = 0.1 * X_sa[t-1] + np.random.normal(0, 1)
        elif 1000 < t <= 1999:
            X_sa[t-1] = 0.4 * X_sa[t-1] + np.random.uniform(-1, 1)
        elif 2000 < t <= 2999:
            X_sa[t-1] = 0.6 * X_sa[t-1] + np.random.laplace(0, 1)
        elif 3000 < t <= 3999:
            X_sa[t-1] = 0.1 * X_sa[t-1] + np.random.normal(0, 1)
        elif 4000 < t <= 4999:
            X_sa[t-1] = 0.4 * X_sa[t-1] + np.random.uniform(-1, 1)
        else:
            X_sa[t-1] = 0.6 * X_sa[t-1] + np.random.laplace(0, 1)
    return X_sa

#%%
def load_SG_data():
    X_sg = np.zeros(5000)
    for t in range(1, 5001):
        if 1 <= t <= 800:
            X_sg[t-1] = 0.1 * X_sg[t-1] + np.random.normal(0, 1)
        elif 800 < t <= 1000:
            X_sg[t-1] = 0.5 * (0.1 * X_sg[t-1] + np.random.normal(0, 1) + 0.4 * X_sg[t-1] + np.random.uniform(-1, 1))
        elif 1000 < t <= 1600:
            X_sg[t-1] = 0.4 * X_sg[t-1] + np.random.uniform(-1, 1)
        elif 1600 < t <= 1800:
            X_sg[t-1] = 0.5 * (0.4 * X_sg[t-1] + np.random.uniform(-1, 1) + 0.6 * X_sg[t-1] + np.random.laplace(0, 1))
        elif 1800 < t <= 2400:
            X_sg[t-1] = 0.6 * X_sg[t-1] + np.random.laplace(0, 1)
        elif 2400 < t <= 2600:
            X_sg[t-1] = 0.5 * (0.6 * X_sg[t-1] + np.random.laplace(0, 1) + 0.1 * X_sg[t-1] + np.random.normal(0, 1))
        elif 2600 < t <= 3200:
            X_sg[t-1] = 0.1 * X_sg[t-1] + np.random.normal(0, 1)
        elif 3200 < t <= 3400:
            X_sg[t-1] = 0.5 * (0.1 * X_sg[t-1] + np.random.normal(0, 1) + 0.4 * X_sg[t-1] + np.random.uniform(-1, 1))
        elif 3400 < t <= 4000:
            X_sg[t-1] = 0.4 * X_sg[t-1] + np.random.uniform(-1, 1)
        elif 4000 < t <= 4200:
            X_sg[t-1] = 0.5 * (0.4 * X_sg[t-1] + np.random.uniform(-1, 1) + 0.6 * X_sg[t-1] + np.random.laplace(0, 1))
        else:
            X_sg[t-1] = 0.6 * X_sg[t-1] + np.random.laplace(0, 1)
    return X_sg
