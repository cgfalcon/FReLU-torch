import torch.nn as nn
import torch
import torch.nn.functional as F


class FReLU(nn.Module):

    def __init__(self, frelu_init = 1.0, inplace = False):
        super(FReLU, self).__init__()
        if frelu_init == 0:
            self.b = nn.Parameter(torch.zeros(1))
        else:
            self.b = nn.Parameter(torch.ones(1) * frelu_init)
        self.inplace = inplace

    def forward(self, x):
        return F.relu(x, inplace=self.inplace) + self.b

    def extra_repr(self) -> str:
        repr = f'inplace={"True" if self.inplace else ""}, bias=({self.b.shape})'
        return repr