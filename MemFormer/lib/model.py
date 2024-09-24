import torch
import torch.nn as nn
import torch.nn.init as init
from MemFormer.lib.DilatedConv import MemTCN

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


class Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder = MemTCN(input_dims=opt.z_dim, output_dims=opt.intermediate_dim, hidden_dims=64, depth=10) 
        self.dim = opt.hidden_dim
        
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=opt.intermediate_dim, nhead=8, batch_first=True), num_layers=2)
        self.fc = nn.Linear(opt.intermediate_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        rep = self.encoder(x)
        y = self.transformer_encoder(rep)
        y = self.fc(y)
        y = self.sigmoid(y)
        return y
    def store_grad(self):
        for name, layer in self.encoder.named_modules():    
            if 'PadConv' in type(layer).__name__:
                layer.store_grad()


class Recovery(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder = MemTCN(input_dims=opt.hidden_dim,
                             output_dims=opt.intermediate_dim,
                             hidden_dims=64, 
                             depth=10) 
        self.dim = opt.z_dim
        
        self.transformer_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=opt.intermediate_dim, nhead=8, batch_first=True), num_layers=2)
        self.fc = nn.Linear(opt.intermediate_dim, opt.z_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        rep = self.encoder(x)
        y = self.transformer_decoder(rep, rep)
        y = self.fc(y)
        y = self.sigmoid(y)
        return y
    def store_grad(self):
        for name, layer in self.encoder.named_modules():    
            if 'PadConv' in type(layer).__name__:
                layer.store_grad()


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.rnn = nn.GRU(input_size=opt.z_dim,
                          hidden_size=opt.hidden_dim, num_layers=opt.num_layer)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, sigmoid=True):
        g_outputs, _ = self.rnn(input)
        E = self.fc(g_outputs)
        if sigmoid:
            E = self.sigmoid(E)
        return E


class Supervisor(nn.Module):
    def __init__(self, opt):
        super(Supervisor, self).__init__()
        self.rnn = nn.GRU(input_size=opt.hidden_dim,
                          hidden_size=opt.hidden_dim, num_layers=opt.num_layer)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        #self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        s_outputs, _ = self.rnn(input)
        S = self.fc(s_outputs)
        if sigmoid:
            S = self.sigmoid(S)
        return S


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(input_size=opt.hidden_dim,
                          hidden_size=opt.hidden_dim, num_layers=opt.num_layer)
        #  self.norm = nn.LayerNorm(opt.hidden_dim)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        #self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        d_outputs, _ = self.rnn(input)
        Y_hat = self.fc(d_outputs)
        if sigmoid:
            Y_hat = self.sigmoid(Y_hat)
        return Y_hat
