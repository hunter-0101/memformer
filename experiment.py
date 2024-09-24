import torch
import numpy as np
from data_loading import load_data, augment_data
import matplotlib.pyplot as plt
from downstream_models import LSTMModel, GRUModel, TransformerModel

#%% evaluating the input dataset by training three different downstream models and return their losss history/metrics
def evaluate_data(train_data, test_data):
    input_dim = train_data[0].shape[1]  # Assuming input_dim is feature_dim
    seq_len = train_data[0].shape[0]
    output_dim = input_dim  # Same dimensionality for input/output in time series

    # Initialize downstream models
    lstm_model = LSTMModel(input_dim, 64, output_dim, seq_len, train_data, test_data)
    gru_model = GRUModel(input_dim, 64, output_dim, seq_len, train_data, test_data)
    transformer_model = TransformerModel(input_dim, 64, output_dim, seq_len, train_data, test_data)

    # Train and evaluate LSTM
    lstm_model.train_model(num_epochs=800)
    lstm_model.evaluate()
    print('LSTM metrics:', lstm_model.metrics)

    # Train and evaluate GRU
    gru_model.train_model(num_epochs=800)
    gru_model.evaluate()
    print('GRU metrics:', gru_model.metrics)

    
    # Train and evaluate Transformer
    transformer_model.train_model(num_epochs=400)
    transformer_model.evaluate()
    print('Transformer metrics:', transformer_model.metrics)
    
    
    return lstm_model.loss_history, gru_model.loss_history, transformer_model.loss_history

    print('Finish downstream model evaluation.')
    
#%% load and train over original data
train_data, test_data = load_data('data/ETTh1.csv', 500)
lstm_loss0 , gru_loss0, transformer_loss0= evaluate_data(train_data, test_data)

#%% train TimeGAN and generate synthetic data
from TimeGAN.options import Options
from TimeGAN.lib.data import load_data as load_data_timegan
from TimeGAN.lib.timegan import TimeGAN

opt = Options().parse()
opt.iteration = 50000
opt.metric_iteration = 200
ori_data = load_data_timegan(opt)
model = TimeGAN(opt, ori_data)
model.train()
model.evaluation()
print(model.metrics)
#%% evaluate TimeGAN_augmented data
train_data_aug1 = np.concatenate((train_data, model.generated_data), axis=0)
lstm_loss1 , gru_loss1, transformer_loss1= evaluate_data(train_data_aug1, test_data)

#%% train MemFormer and generate synthetic data
from MemFormer.options import Options
from MemFormer.lib.data import load_data as load_data_timegan
from MemFormer.lib.memformer import MemFormer

opt = Options().parse()
opt.iteration = 50000
opt.metric_iteration = 200
ori_data = load_data_timegan(opt)
model = MemFormer(opt, ori_data)
model.train()
model.evaluation()
print(model.metrics)
train_data_aug2 = np.concatenate((train_data, model.generated_data), axis=0)
lstm_loss2 , gru_loss2, transformer_loss2= evaluate_data(train_data_aug2, test_data)

#%% evaluate MemFormer_augmented data
train_data_aug2 = np.concatenate((train_data, synthetic_data), axis=0)
lstm_loss2 , gru_loss2, transformer_loss2= evaluate_data(train_data_aug2, test_data)
    
#%% plot the loss curve
x = range(1, 801, 1)
plt.figure()
plt.plot(x, lstm_loss0, label='original')
plt.plot(x, lstm_loss1, label='TimeGAN_aug')
plt.plot(x, lstm_loss2, label='MemFormer_aug')
plt.title('LSTM')
plt.legend()

plt.figure()
plt.plot(x, gru_loss0, label='original')
plt.plot(x, gru_loss1, label='TimeGAN_aug')
plt.plot(x, gru_loss2, label='Memformer_aug')
plt.title('GRU')
plt.legend()

x = range(1, 401, 1)
plt.figure()
plt.plot(x, transformer_loss0, label='original')
plt.plot(x, transformer_loss1, label='TimeGAN_aug')
plt.plot(x, transformer_loss2, label='MemFormer_aug')
plt.title('Transformer')
plt.legend()

plt.show()
