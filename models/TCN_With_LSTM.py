import torch.nn as nn
import torch

import models.TemporalConvCell as TCN

class TCN_LSTM(nn.Module):
    def __init__(self, in_channels, conv_channels, num_blocks, spatial_kernel, temporal_kernel, dilation, lstm_hidden_size, lstm_layers, num_classes):
        super().__init__()

        self.tcn = TCN.TemporalConvCell(in_channels, conv_channels, spatial_kernel, temporal_kernel, dilation)
        self.lstm = nn.LSTM(conv_channels, lstm_hidden_size, lstm_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

