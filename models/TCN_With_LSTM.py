import torch.nn as nn
import torch

import models.TemporalConvCell as TCN

class TCN_LSTM(nn.Module):
    def __init__(self, 
                 in_channels, conv_channels, num_blocks,
                 spatial_kernel, temporal_kernel,
                 lstm_hidden, lstm_layers, num_classes,
                 input_size=(64, 64), dilation_rates=None):

        super().__init__()
        self.num_blocks = num_blocks

        if dilation_rates is None:
            dilation_rates = [2 ** i for i in range(num_blocks)]

        self.temp_conv_blocks = nn.ModuleList()
        for i in range(num_blocks):
            in_ch = in_channels if i == 0 else conv_channels[min(i-1, len(conv_channels)-1)]
            out_ch = conv_channels[min(i, len(conv_channels)-1)]
            self.temp_conv_blocks.append(
                TCN.TemporalConvCell(in_ch, out_ch, spatial_kernel, temporal_kernel, dilation=dilation_rates[i])
            )

        self.maxpool = nn.AdaptiveMaxPool2d((16,16))
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.num_classes = num_classes

        self.lstm = None
        self.fc = None

    def initialize_lstm(self, sample):
        if self.lstm is None:
            with torch.no_grad():
                x = sample.permute(0, 2, 1, 3, 4)
                for block in self.temp_conv_blocks:
                    x = block(x)
                b, c, t, h, w = x.shape
                x = x.permute(0, 2, 1, 3, 4)
                x_pooled = [self.maxpool(x[:, i]) for i in range(t)]
                x = torch.stack([p.reshape(b, -1) for p in x_pooled], dim=1)
                feature_dim = x.shape[-1]
            self.lstm = nn.LSTM(
                input_size=feature_dim,
                hidden_size=self.lstm_hidden,
                num_layers=self.lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.3 if self.lstm_layers > 1 else 0
            )
            self.fc = nn.Linear(self.lstm_hidden * 2, self.num_classes)

            self.to(next(self.parameters()).device)



    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        for block in self.temp_conv_blocks:
            x = block(x)

        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x_pooled = [self.maxpool(x[:, i]) for i in range(t)]
        x = torch.stack([p.reshape(b, -1) for p in x_pooled], dim=1)

        rnn_out, _ = self.lstm(x)
        last_out = rnn_out[:, -1, :]
        return self.fc(last_out)
    
if __name__ == "__main__":
    model = TCN_LSTM(in_channels=3, conv_channels=[64, 128], num_blocks=2, spatial_kernel=3, temporal_kernel=3, lstm_hidden=256, lstm_layers=2, input_size=(128, 128), num_classes=8)
    dummy_input = torch.randn(2, 30, 3, 128, 128)

    # try:
    output = model(dummy_input)
    print("Output shape:", output.shape)  
    # except Exception as e:
    #     print("Error occurred while testing the model:", e)