import torch 

class RMSNorm(torch.nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):

        super().__init__()
        self.eps    = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x:torch.Tensor) -> torch.Tensor:
        
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
def default_initialize(layernorm_id: str, dim_sz: int) -> torch.nn.Module:

    if layernorm_id == "rmsnorm":
        return RMSNorm(dim_sz)
    
    if layernorm_id == "std":
        return torch.nn.LayerNorm(dim_sz)
    
    raise Exception()
