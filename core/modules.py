import torch
from torch import nn
import numpy as np


class SettledPositionalEncoding(nn.Module):

    def __init__(self, d_hid: int, n_position: int=200) -> None:
        super(SettledPositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x: [batch, len, dim] float32
            return: [batch, len, dim] float32
        '''
        return x + self.pos_table[:, :x.size(1)].clone().detach()
