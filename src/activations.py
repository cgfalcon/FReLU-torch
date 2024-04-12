import torch.nn as nn
import torch
import torch.nn.functional as F


class FReLU(nn.Module):

    def __init__(self, frelu_init = 1, inplace = False):
        super(FReLU, self).__init__()
        if frelu_init == 0:
            self.b = nn.Parameter(torch.zeros(1))
        else:
            self.b = nn.Parameter(torch.ones(1))
        self.inplace = inplace

    def forward(self, x):
        return F.relu(x, inplace=self.inplace) + self.b

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str