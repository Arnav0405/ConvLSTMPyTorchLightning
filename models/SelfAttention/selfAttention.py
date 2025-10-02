import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Conv2d(input_dim, hidden_dim, 1)
        self.key = nn.Conv2d(input_dim, hidden_dim, 1)
        self.value = nn.Conv2d(input_dim, input_dim, 1)
        self.z = nn.Conv2d(input_dim, input_dim, 1)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, h):
        batch_size, _, H, W = h.shape
        k_h = self.key(h)
        q_h = self.query(h)
        v_h = self.value(h)

        k_h = k_h.view(batch_size, self.hidden_dim, H * W)
        q_h = q_h.view(batch_size, self.hidden_dim, H * W).transpose(1, 2)
        v_h = v_h.view(batch_size, self.input_dim, H * W)

        attention = torch.softmax(torch.bmm(q_h, k_h), dim=-1) 

        new_h = torch.matmul(attention, v_h.permute(0, 2, 1))
        new_h = new_h.transpose(1, 2).view(batch_size, self.input_dim, H, W)
        new_h = self.z(new_h)

        return new_h, attention