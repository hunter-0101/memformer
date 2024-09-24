import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def generate_batch(data, batch_size=128):
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]
    
    X = [data[i] for i in train_idx]
    
    return X
    
# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, train_data, test_data, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.seq_len = seq_len
        self.train_data = train_data
        self.test_data = test_data
        self.loss_history = []
        self.metrics = {}

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

    def train_model(self, num_epochs=100, lr=0.001):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0
            input = torch.tensor(generate_batch(self.train_data, 128), dtype=torch.float32)
            optimizer.zero_grad()
            output = self(input)
            loss = criterion(output, input[:, -1, :])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            self.loss_history.append(epoch_loss / len(input))

    def evaluate(self):
        self.eval()
        mse = nn.MSELoss()
        mae = nn.L1Loss()
        mse_loss, mae_loss = 0, 0
        with torch.no_grad():
            input = torch.tensor(self.test_data,dtype=torch.float32)
            output = self(input)
            mse_loss += mse(output, input[:, -1, :]).item()
            mae_loss += mae(output, input[:, -1, :]).item()
        self.metrics['MSE'] = mse_loss / len(self.test_data)
        self.metrics['MAE'] = mae_loss / len(self.test_data)


# GRU Model (similar structure to LSTM)
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, train_data, test_data, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.seq_len = seq_len
        self.train_data = train_data
        self.test_data = test_data
        self.loss_history = []
        self.metrics = {}

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

    def train_model(self, num_epochs=100, lr=0.001):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0
            input = torch.tensor(generate_batch(self.train_data, 128), dtype=torch.float32)
            optimizer.zero_grad()
            output = self(input)
            loss = criterion(output, input[:, -1, :])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            self.loss_history.append(epoch_loss / len(input))

    def evaluate(self):
        self.eval()
        mse = nn.MSELoss()
        mae = nn.L1Loss()
        mse_loss, mae_loss = 0, 0
        with torch.no_grad():
            input = torch.tensor(self.test_data, dtype=torch.float32) 
            output = self(input)
            mse_loss += mse(output, input[:, -1, :]).item()
            mae_loss += mae(output, input[:, -1, :]).item()
        self.metrics['MSE'] = mse_loss / len(self.test_data)
        self.metrics['MAE'] = mae_loss / len(self.test_data)


# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, train_data, test_data, num_layers=2, num_heads=1):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(input_dim, num_heads, num_layers, num_layers)
        self.fc = nn.Linear(seq_len * input_dim, output_dim)
        self.seq_len = seq_len
        self.train_data = train_data
        self.test_data = test_data
        self.loss_history = []
        self.metrics = {}

    def forward(self, x):
        x = x.permute(1, 0, 2)  # For transformer input (seq_len, batch, input_dim)
        out = self.transformer(x, x)
        out = out.permute(1, 0, 2).contiguous().view(x.size(1), -1)
        out = self.fc(out)
        return out

    def train_model(self, num_epochs=100, lr=0.001):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0
            input = torch.tensor(generate_batch(self.train_data, 128), dtype=torch.float32)
            optimizer.zero_grad()
            output = self(input)
            loss = criterion(output, input[:, -1, :])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            self.loss_history.append(epoch_loss / len(input))

    def evaluate(self):
        self.eval()
        mse = nn.MSELoss()
        mae = nn.L1Loss()
        mse_loss, mae_loss = 0, 0
        with torch.no_grad():
            input = torch.tensor(self.test_data, dtype=torch.float32)
            output = self(input)
            mse_loss += mse(output, input[:, -1, :]).item()
            mae_loss += mae(output, input[:, -1, :]).item()
        self.metrics['MSE'] = mse_loss / len(self.test_data)
        self.metrics['MAE'] = mae_loss / len(self.test_data)
