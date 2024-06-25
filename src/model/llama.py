import torch
from torch import nn 
from rotary_embedding_torch import RotaryEmbedding 
import math 

MAX_ATTENTION_HEAD  = 8
ATTN_EMB_SIZE       = [256, 512, 1024, 2048]
MLP_EMB_SIZE        = [256, 512, 1024] 

def round_emb(emb_size: int, arr: list[int]) -> int:
    
    for candidate in arr:
        if candidate >= emb_size:
            return candidate
    
    return arr[-1]

def round_attn_emb(emb_size: int) -> int:

    return round_emb(emb_size, ATTN_EMB_SIZE)

def round_mlp_emb(emb_size: int) -> int:

    return round_emb(emb_size, MLP_EMB_SIZE) 

def is_pow_2(x: int) -> bool: 
    
    return x > 0 and (x & (x - 1)) == 0 

class RMSNorm(torch.nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):

        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x:torch.Tensor) -> torch.Tensor:
        
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

#rearrange ctx points.
#newer models aim to fix the ctx arrangement problem by doing more row linear and transpose 
#this speaks to the core weakness of transformer architecture - context arrangement problem 
class ContextMixer(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        _, seq_sz, ctx_sz = x.size()

        x = torch.topk(x, ctx_sz)[0]
        x = x.transpose(1, 2)
        x = torch.topk(x, seq_sz)[0]
        x = x.transpose(1, 2)

        return x

class RowLinear(nn.Module):

    def __init__(self, *args, **kwargs):

        super().__init__()
        self.ln = nn.Linear(*args, **kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_sz, seq_len, dim = x.size()
        x   = x.transpose(1, 2)
        x   = x.contiguous().view(batch_sz, seq_len, dim)
        x   = self.ln(x)
        _, _, new_dim = x.size()
        x   = x.contiguous().view(batch_sz, new_dim, -1)
        x   = x.transpose(1, 2)

        return x

class Attention(nn.Module):

    def __init__(self, emb_size: int, scale_ratio: int, head_count: int):
        
        assert head_count <= MAX_ATTENTION_HEAD and is_pow_2(head_count) 

        super().__init__()

        self.proj_embed     = round_attn_emb(emb_size * scale_ratio)
        self.rot            = RotaryEmbedding(self.proj_embed // head_count)
        self.ln1            = nn.Linear(emb_size, self.proj_embed, bias = False)
        self.ln2            = nn.Linear(emb_size, self.proj_embed, bias = False)
        self.ln3            = nn.Linear(emb_size, self.proj_embed, bias = False)
        self.rln            = RowLinear(self.proj_embed, emb_size, bias = False)
        self.n_head         = head_count
    
    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        
        batch_sz, seq_len, _    = emb.size()
        
        q: torch.Tensor = self.ln1(emb)
        k: torch.Tensor = self.ln2(emb)
        v: torch.Tensor = self.ln3(emb)
        
        q   = q.contiguous().view(batch_sz, self.n_head, seq_len, -1)
        k   = k.contiguous().view(batch_sz, self.n_head, seq_len, -1)
        v   = v.contiguous().view(batch_sz, self.n_head, seq_len, -1)

        q   = self.rot.rotate_queries_or_keys(q)
        k   = self.rot.rotate_queries_or_keys(k)

        scr = torch.softmax(torch.matmul(q, k.transpose(2, 3)).float() / math.sqrt(self.n_head), dim=-1).type_as(v)
        out = torch.matmul(scr, v)
        out = out.transpose(1,2).contiguous().view(batch_sz, seq_len, -1) #

        return self.rln(out)

class MLP(nn.Module):

    def __init__(self, emb_size: int, scale_ratio: int):
        
        super().__init__()

        self.act        = nn.SiLU()
        self.w1         = nn.Linear(emb_size, round_mlp_emb(emb_size * scale_ratio), bias = False)
        self.w2         = RowLinear(round_mlp_emb(emb_size * scale_ratio), emb_size, bias = False)
        self.w3         = nn.Linear(emb_size, round_mlp_emb(emb_size * scale_ratio), bias = False)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:

        return self.w2(self.act(self.w1(emb)) * self.w3(emb))

class Block(nn.Module):

    def __init__(self, emb_size: int, attn_scale_ratio: int, attn_head_count: int, mlp_scale_ratio: int):

        super().__init__()

        self.attn   = Attention(emb_size, attn_scale_ratio, attn_head_count)
        self.mlp    = MLP(emb_size, mlp_scale_ratio)
        self.lnorm1 = RMSNorm(emb_size)
        self.lnorm2 = RMSNorm(emb_size)
    
    def forward(self, emb: torch.Tensor) -> torch.Tensor:

        emb = emb + self.attn(self.lnorm1(emb))
        emb = emb + self.mlp(self.lnorm2(emb))
        
        return emb 

class GPT(nn.Module): 

    def __init__(self, vocab_size: int, emb_size: int, attn_scale_ratio: int, attn_head_count: int, mlp_scale_ratio: int, block_sz: int):
        
        super().__init__()

        self.emb        = nn.Embedding(vocab_size, emb_size)
        self.blocks     = [Block(emb_size, attn_scale_ratio, attn_head_count, mlp_scale_ratio) for _ in range(block_sz)]
        self.ln_out     = nn.Linear(emb_size, vocab_size)
        self.lnorm      = RMSNorm(emb_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        
        hid = self.emb(sentence)

        for block in self.blocks:
            hid = block(hid)

        return self.ln_out(self.lnorm(hid))