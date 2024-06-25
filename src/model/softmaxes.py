import torch 

i = 0

class StdSoftmax(torch.nn.Module):

    def __init__(self): 

        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.softmax(x, dim=-1)

class HyperSoftmax(torch.nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return torch.exp(x) / (torch.sum(torch.exp(x), -1, keepdim=True) + 1)