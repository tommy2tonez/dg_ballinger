import torch 
from typing import Union 

class Serf(torch.nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor: 

        return x * torch.erf(torch.log(1 + torch.exp(x)))

class SinSig(torch.nn.Module):

    def __init__(self):

        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x * torch.sin(torch.pi / 2 * torch.special.expit(x))

class TSwish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor):

        ctx.save_for_backward(x)

        neg_idx: torch.Tensor       = x < 0
        non_neg_idx: torch.Tensor   = x >= 0
        rs: torch.Tensor            = x.clone()
        rs[neg_idx]                 = -0.20 
        rs[non_neg_idx]             = (x[non_neg_idx] / (1 + torch.exp(-x[non_neg_idx])) -0.20).type_as(x)

        return rs

    @staticmethod 
    def backward(ctx, grad_x: torch.Tensor):
        
        x, = ctx.saved_tensors
        
        grad_x_out = grad_x.clone()
        grad_x_out[x < 0] = 0

        return grad_x_out

class FlattenedTSwish(torch.nn.Module):

    def __init__(self):

        super().__init__()
        self.tswish = TSwish() 

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.nn.functional.relu(x) * torch.special.expit(x) + self.tswish.apply(x) 

class RSigELUD(torch.nn.Module):

    def __init__(self, a: float = 0.5, b: float = 0.5):

        super().__init__()
        self.a = a 
        self.b = b
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        neg_idx: torch.Tensor   = x < 0
        pos_idx: torch.Tensor   = x > 1
        rs: torch.Tensor        = x.clone()
        rs[neg_idx]             = (self.b * (torch.exp(x[neg_idx]) - 1)).type_as(x)
        rs[pos_idx]             = (x[pos_idx] * (1 / (1 + torch.exp(-x[pos_idx]))) * self.a + x[pos_idx]).type_as(x)

        return rs

class AReLU(torch.nn.Module):

    def __init__(self, alpha=0.90, beta=2.0):

        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1) * alpha)
        self.beta = torch.nn.Parameter(torch.ones(1) * beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        alpha   = torch.clamp(self.alpha, min=0.01, max=0.99)
        beta    = 1 + torch.sigmoid(self.beta)

        return torch.nn.functional.relu(x) * beta - torch.nn.functional.relu(-x) * alpha

class ChPAF(torch.nn.Module):

    def __init__(self, dim_sz: int, poly_order_count: int = 3):
        
        super().__init__()
        self.a = [torch.nn.Parameter(torch.ones(dim_sz)) for _ in range(poly_order_count)]
        self.poly_order_count: int = poly_order_count

    def c(self, x: torch.Tensor, i: int) -> Union[torch.Tensor, int]:

        if i == 0:
            return 1
        
        if i == 1:
            return x 
        
        return 2*x*self.c(x, i-1) - self.c(x, i-2) 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return sum([self.a[i] * self.c(x, i) for i in range(self.poly_order_count)])

class Gish(torch.nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x * torch.log(2 - torch.exp(-torch.exp(x)))

class Phish(torch.nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x * torch.tanh(torch.nn.functional.gelu(x)) 

class ESwish(torch.nn.Module):

    def __init__(self, dim_sz: int):

        super().__init__()
        self.params = torch.nn.Parameter(torch.ones(dim_sz))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.clamp(self.params, 1, 2) * x * torch.special.expit(x)

class PSGU(torch.nn.Module):

    def __init__(self, dim_sz: int):

        super().__init__()
        self.params = torch.nn.Parameter(torch.ones(dim_sz) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x * torch.tanh(self.params * torch.special.expit(x)) 

class BLU(torch.nn.Module):

    def __init__(self, dim_sz: int):

        super().__init__()
        self.params = torch.nn.Parameter(torch.ones(dim_sz))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.clamp(self.params, -1, 1) * (torch.sqrt(torch.pow(x, 2) + 1) - 1) + x
    
class AppSquaredReLu(torch.nn.Module):

    def __init__(self, dim: int):

        super().__init__()
        self.w_up       = torch.nn.Parameter(torch.ones(dim))
        self.w_down     = torch.nn.Parameter(torch.ones(dim))
        self.dim        = dim
         
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return torch.pow(torch.relu(x * self.w_up), 2) * self.w_down 
    
def default_initialize(act_id: str, dim_sz: int) -> torch.nn.Module:
    
    if act_id.lower() == "serf":
        return Serf()
    
    if act_id.lower() == "sinsig":
        return SinSig()
    
    if act_id.lower() == "flattenedtswish":
        return FlattenedTSwish()
    
    if act_id.lower() == "rsigelud":
        return RSigELUD()
    
    if act_id.lower() == "arelu":
        return AReLU()
    
    if act_id.lower() == "chpaf":
        return ChPAF(dim_sz)
    
    if act_id.lower() == "gish":
        return Gish()
    
    if act_id.lower() == "phish":
        return Phish()
    
    if act_id.lower() == "eswish":
        return ESwish(dim_sz)
    
    if act_id.lower() == "psgu":
        return PSGU(dim_sz)
    
    if act_id.lower() == "blu":
        return BLU(dim_sz)
    
    if act_id.lower() == "appsquaredrelu":
        return AppSquaredReLu(dim_sz)
    
    if act_id.lower() == "swish":
        return torch.nn.SiLU()

    if act_id.lower() == "hardswish":
        return torch.nn.Hardswish() 
    
    raise Exception()