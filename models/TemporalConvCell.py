import torch.nn as nn


class Chomp1d(nn.Module):
    """
    Removes the trailing padding from the temporal dimension to ensure causality
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalConvCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, temporal_kernel=3, dilation=1):
        super(TemporalConvCell, self).__init__()
        
        self.spatial_conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            padding=kernel_size//2
        )

        self.temporal_conv = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=temporal_kernel,
            padding=temporal_kernel//2
        )
        
        self.temporal_padding = (temporal_kernel - 1) * dilation
        self.chomp = Chomp1d(self.temporal_padding)

        self.bn_spatial = nn.BatchNorm2d(out_channels)
        self.bn_temporal = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout()

    def forward(self, x):
        b, c, t, h, w = x.shape
        
        x_reshaped = x.reshape(b * t, c, h, w)
        x_spatial = self.spatial_conv(x_reshaped)
        x_spatial = self.bn_spatial(x_spatial)
        x_spatial = self.relu(x_spatial)
        
        _, c_out, h_out, w_out = x_spatial.shape
        x_spatial = x_spatial.reshape(b, t, c_out, h_out, w_out)
        x_spatial = x_spatial.permute(0, 2, 1, 3, 4)  # (b, c, t, h, w)
        
        x_temp = x_spatial.reshape(b, c_out, t, h_out * w_out)
        x_temp = x_temp.permute(0, 3, 1, 2)  # (b, h*w, c, t)
        x_temp = x_temp.reshape(b * h_out * w_out, c_out, t)    # (b*h*w, c, t)
        x_temp = self.temporal_conv(x_temp)
        x_temp = self.chomp(x_temp)

        t_out = x_temp.size(2)

        x_temp = self.bn_temporal(x_temp)
        x_temp = self.relu(x_temp)
        x_temp = self.drop(x_temp)
        x_temp = x_temp.reshape(b, h_out * w_out, c_out, t_out)
        x_temp = x_temp.permute(0, 2, 3, 1)  # (b, c, t, h*w)
        x_temp = x_temp.reshape(b, c_out, t_out, h_out, w_out)
        
        return x_temp