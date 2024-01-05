import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    
    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src) #Encoder memory
        output = self.gelu(output)
        output = self.decoder(output) #Linear layer
        output = self.gelu(output)
        output = self.flatten(output)
        v = self.v_output(output) #Value output
        v = self.softmax(v) #Get softmax probability
        p = self.p_output(output) #Policy output
        p = self.softmax(p) #Get softmax probability
        return v, p
    
    def __init__(self, sinp, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, padding_idx=32):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout) #Positional encoding layer
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout) #Encoder layers
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers) #Wrap all encoder nodes (multihead)
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=padding_idx) #Initial encoding of imputs embed layers
        self.padding_idx = padding_idx #Index of padding token
        self.ninp = ninp #Number of input items
        self.softmax = nn.Softmax(dim=1) #Softmax activation layer
        self.gelu = nn.GELU() #GELU activation layer
        self.flatten = nn.Flatten(start_dim=1) #Flatten layer
        self.decoder = nn.Linear(ninp,1) #Decode layer
        self.v_output = nn.Linear(sinp,3) #Decode layer
        self.p_output = nn.Linear(sinp,4096) #Decode layer
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
    
    def get_batch(source, x, y):
        data = torch.tensor([])
        v_target = torch.tensor([])
        p_target = torch.tensor([])
        for i in range(y):
            #Training data
            if len(source) > 0 and x+i < len(source):
                d_seq = source[x+i][:len(source[x+i])-4099]
                data = torch.cat((data, d_seq))
                #Target data
                v_seq = source[x+i][-3:]
                v_target = torch.cat((v_target, v_seq))
                p_seq = source[x+i][-4099:-3]
                p_target = torch.cat((p_target, p_seq))
        return data.reshape(min(y, len(source[x:])), len(source[0])-4099).to(torch.int64), v_target.reshape(min(y, len(source[x:])), 3).to(torch.float), p_target.reshape(min(y, len(source[x:])), 4096).to(torch.float)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


