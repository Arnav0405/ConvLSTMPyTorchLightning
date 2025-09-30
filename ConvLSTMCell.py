import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(self.input_dim + self.hidden_dim, out_channels= 4 * self.hidden_dim,
                                  kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)
        

    def forward(self, input_tensor, cur_state):
        h_curr, c_curr = cur_state

        combined = torch.cat([input_tensor, h_curr], dim=1)
        combined_conv = self.conv(combined)

        return 