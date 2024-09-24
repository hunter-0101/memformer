import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from TimeGAN.utils import extract_time

def predictive_score_metrics(ori_data, generated_data):
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    # Define the RNN-based predictive network
    class Predictor(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(Predictor, self).__init__()
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x, lengths):
            # Pack the sequences for the RNN
            packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.rnn(packed_x)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            y_hat = self.fc(out)
            return y_hat

    hidden_dim = int(dim/2)
    iterations = 500
    batch_size = 128

    predictor = Predictor(dim - 1, hidden_dim)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(predictor.parameters())

    # Training using the synthetic dataset
    for itt in range(iterations):
        # Set mini-batch
        idx = np.random.permutation(len(generated_data))
        train_idx = idx[:batch_size]

        X_mb = torch.tensor([generated_data[i][:-1, :(dim-1)] for i in train_idx], dtype=torch.float32)
        T_mb = torch.tensor([generated_time[i]-1 for i in train_idx], dtype=torch.long)
        Y_mb = torch.tensor([np.reshape(generated_data[i][1:, (dim-1)], [-1, 1]) for i in train_idx], dtype=torch.float32)

        # Train predictor
        optimizer.zero_grad()
        y_pred = predictor(X_mb, T_mb)
        p_loss = criterion(y_pred, Y_mb)
        p_loss.backward()
        optimizer.step()

    # Test the trained model on the original data
    idx = np.random.permutation(len(ori_data))
    test_idx = idx[:no]

    X_mb = torch.tensor([ori_data[i][:-1, :(dim-1)] for i in test_idx], dtype=torch.float32)
    T_mb = torch.tensor([ori_time[i]-1 for i in test_idx], dtype=torch.long)
    Y_mb = torch.tensor([np.reshape(ori_data[i][1:, (dim-1)], [-1, 1]) for i in test_idx], dtype=torch.float32)

    with torch.no_grad():
        pred_Y_curr = predictor(X_mb, T_mb)

    # Compute the performance in terms of MAE
    MAE_temp = 0
    for i in range(no):
        MAE_temp += torch.mean(torch.abs(Y_mb[i] - pred_Y_curr[i]))

    predictive_score = MAE_temp.item() / no

    return predictive_score
