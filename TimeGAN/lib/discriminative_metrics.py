import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from TimeGAN.utils import train_test_divide, extract_time
from TimeGAN.lib.data import batch_generator

def discriminative_score_metrics(ori_data, generated_data):
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    # Define the RNN-based discriminator network
    class Discriminator(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(Discriminator, self).__init__()
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x, lengths):
            # Pack the sequences for the RNN
            packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.rnn(packed_x)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            last_outputs = out[torch.arange(out.size(0)), lengths - 1]
            y_hat_logit = self.fc(last_outputs)
            y_hat = torch.sigmoid(y_hat_logit)
            return y_hat_logit, y_hat

    hidden_dim = int(dim/2)
    iterations = 200 
    batch_size = 128

    discriminator = Discriminator(dim, hidden_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(discriminator.parameters())

    # Train/test division for both original and generated data
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)

    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_x_hat = torch.tensor(test_x_hat, dtype=torch.float32)    
    test_t = torch.tensor(test_t, dtype=torch.long)
    test_t_hat = torch.tensor(test_t_hat, dtype=torch.long)
    
    # Training step
    for itt in range(iterations):
        # Batch setting
        X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)

        # Convert batches to PyTorch tensors
        X_mb = torch.tensor(X_mb, dtype=torch.float32)
        T_mb = torch.tensor(T_mb, dtype=torch.long)
        X_hat_mb = torch.tensor(X_hat_mb, dtype=torch.float32)
        T_hat_mb = torch.tensor(T_hat_mb, dtype=torch.long)

        # Train discriminator
        optimizer.zero_grad()

        y_logit_real, _ = discriminator(X_mb, T_mb)
        y_logit_fake, _ = discriminator(X_hat_mb, T_hat_mb)

        d_loss_real = criterion(y_logit_real, torch.ones_like(y_logit_real))
        d_loss_fake = criterion(y_logit_fake, torch.zeros_like(y_logit_fake))
        d_loss = d_loss_real + d_loss_fake

        d_loss.backward()
        optimizer.step()

    # Test the performance on the testing set
    with torch.no_grad():
        _, y_pred_real = discriminator(test_x, test_t)
        _, y_pred_fake = discriminator(test_x_hat, test_t_hat)

    y_pred_final = torch.cat([y_pred_real, y_pred_fake], dim=0).squeeze().numpy()
    y_label_final = np.concatenate((np.ones(len(y_pred_real)), np.zeros(len(y_pred_fake))))

    # Compute the accuracy
    acc = np.mean((y_pred_final > 0.5) == y_label_final)
    discriminative_score = np.abs(0.5 - acc)

    return discriminative_score
