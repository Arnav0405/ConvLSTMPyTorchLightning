import torch.nn as nn
import torch

import models.TemporalConvCell as TCN

class TCN_LSTM(nn.Module):
    def __init__(self, 
                 in_channels, 
                 conv_channels, 
                 num_blocks, 
                 spatial_kernel, 
                 temporal_kernel,  
                 lstm_hidden, lstm_layers, num_classes, input_size=(64, 64), dilation_rates=None):

        super().__init__()

        self.num_blocks = num_blocks

        if dilation_rates is None:
            dilation_rates = [2 ** i for i in range(num_blocks)]

        self.temp_conv_blocks = nn.ModuleList()

        for i in range(num_blocks):
            in_ch = in_channels if i == 0 else conv_channels[min(i-1, len(conv_channels)-1)]
            out_ch = conv_channels[min(i, len(conv_channels)-1)]
            dilation = dilation_rates[i]
            
            self.temp_conv_blocks.append(
                TCN.TemporalConvCell(in_ch, out_ch, spatial_kernel, temporal_kernel, 
                                dilation=dilation)
            )
        
        self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        final_channels = conv_channels[-1]

        self.lstm = nn.LSTM(input_size=final_channels,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.3 if lstm_layers > 1 else 0)

        self.fc = nn.Linear(lstm_hidden * 2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        for block in self.temp_conv_blocks:
            x = block(x)
        
        b, c, t, h, w = x.shape
        # print("Time: ", t)
        # print("Before max pooling", x.shape)
        # Apply max pooling per frame
        x = x.permute(0, 2, 1, 3, 4)  # (b, t, c, h, w)
        x_pooled = []
        for i in range(t):
            frame = x[:, i, :, :, :]  # (b, c, h, w)
            pooled = self.maxpool(frame)  # (b, c, 1, 1)
            pooled = pooled.squeeze(-1).squeeze(-1)  # (b, c)
            x_pooled.append(pooled)
        
        x = torch.stack(x_pooled, dim=1)  # (b, t, c)
        # print("After max pooling", x.shape)
        
        # Bidirectional RNN
        rnn_out, _ = self.lstm(x)  # (b, t, hidden*2)
        
        last_out = rnn_out[:, -1, :]  # (b, hidden*2)
        
        # Classification
        out = self.fc(last_out)  # (b, num_classes)
        out = self.softmax(out)  # (b, num_classes)
        
        return out