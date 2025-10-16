import torch.nn as nn
import torch
from models.SelfAttention.selfAttention import SelfAttention
from models.ConvLSTMCell import ConvLSTMCell

class SAConvLstmCell(nn.Module):
    def __init__ (self, attention_hidden_dims, in_channels, out_channels, kernel_size, padding):
        super(SAConvLstmCell, self).__init__()
        self.attention_x = SelfAttention(in_channels, attention_hidden_dims)
        self.attention_h = SelfAttention(out_channels, attention_hidden_dims)
        self.conv_lstm_cell = ConvLSTMCell(input_dim=in_channels,
                                      hidden_dim=out_channels,
                                      kernel_size=kernel_size,
                                      padding=padding)
    
    def forward(self, x, h, c):
        X, _ = self.attention_x(x)
        new_h, new_c = self.conv_lstm_cell(X, h, c)
        new_h, atten = self.attention_h(new_h)
        new_h += new_h

        return new_h, new_c, atten