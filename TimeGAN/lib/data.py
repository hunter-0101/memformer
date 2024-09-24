import numpy as np
from os.path import dirname, abspath


def MinMaxScaler(data):

    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def sine_data_generation(no, seq_len, dim):

    # Initialize the output
    data = list()

    # Generate sine data
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1)*0.5
        # Stack the generated data
        data.append(temp)

    return data


def real_data_loading(data_name, seq_len):

    assert data_name in ['stock', 'energy', 'ETTh1']

    if data_name == 'stock':
        ori_data = np.loadtxt(dirname(
            dirname(abspath(__file__))) + '/data/stock_data.csv', delimiter=",", skiprows=1)
    elif data_name == 'energy':
        ori_data = np.loadtxt(dirname(dirname(
            abspath(__file__))) + '/data/energy_data.csv', delimiter=",", skiprows=1)
    elif data_name == 'ETTh1':
        ori_data = np.loadtxt(dirname(
            dirname(abspath(__file__))) + '/data/ETTh1.csv', delimiter=',', skiprows=1)

    # Flip the data to make chronological data
    ori_data = ori_data[::-1]
    # Normalize the data
    ori_data = MinMaxScaler(ori_data)

    # Preprocess the dataset
    temp_data = []
    # Cut original dataset into trainnig instances (small lists of seq_len)
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Shuffle the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    return data # this output is a shuffled, train-ready dataset consisting of seq_len instances


def load_data(opt):
    # Data loading
    if opt.data_name in ['stock', 'energy', 'ETTh1']:
        ori_data = real_data_loading(
            opt.data_name, opt.seq_len)  # list: 3661; [24,6]
    elif opt.data_name == 'sine':
        # Set number of samples and its dimensions
        no, dim = 10000, 5
        ori_data = sine_data_generation(no, opt.seq_len, dim)
    print(opt.data_name + ' dataset is ready.')

    return ori_data


def batch_generator(data, time, batch_size):

    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = [data[i] for i in train_idx]
    T_mb = [time[i] for i in train_idx]

    return X_mb, T_mb