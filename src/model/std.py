import torch
from torch import nn 
from rotary_embedding_torch import RotaryEmbedding 
import math 
from . import activators
from . import layernorms
from . import softmaxes

MAX_ATTENTION_HEAD  = 16
ATTN_EMB_SIZE       = [256, 512, 1024, 2048, 4096, 8192]
MLP_EMB_SIZE        = [256, 512, 1024, 2048, 4096, 8192] 

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

    def __init__(self, emb_size: int, qk_scale_ratio: int, qk_rot_perc: float, v_scale_ratio: int, head_count: int):
        
        assert head_count <= MAX_ATTENTION_HEAD and is_pow_2(head_count) 

        super().__init__()

        self.n_head     = head_count
        qk_proj_sz      = round_attn_emb(emb_size * qk_scale_ratio)
        v_proj_sz       = round_attn_emb(emb_size * v_scale_ratio)
        rot_head_dim    = int(qk_proj_sz // head_count * qk_rot_perc) 
        self.rot        = RotaryEmbedding(dim = rot_head_dim) 
        self.qln        = nn.Linear(emb_size, qk_proj_sz, bias = False)
        self.kln        = nn.Linear(emb_size, qk_proj_sz, bias = False)
        self.vln        = nn.Linear(emb_size, v_proj_sz, bias = False)
        self.rln        = RowLinear(v_proj_sz, emb_size, bias = False)
        self.csm        = softmaxes.HyperSoftmax()

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        
        batch_sz, seq_len, _    = emb.size()
        q: torch.Tensor         = self.qln(emb)
        k: torch.Tensor         = self.kln(emb)
        v: torch.Tensor         = self.vln(emb)
        
        q   = q.contiguous().view(batch_sz, self.n_head, seq_len, -1)
        k   = k.contiguous().view(batch_sz, self.n_head, seq_len, -1)
        v   = v.contiguous().view(batch_sz, self.n_head, seq_len, -1)

        q   = self.rot.rotate_queries_or_keys(q)
        k   = self.rot.rotate_queries_or_keys(k)

        scr = self.csm(torch.matmul(q, k.transpose(2, 3)).float() / math.sqrt(self.n_head)).type_as(v)
        out = torch.matmul(scr, v)
        out = out.transpose(1,2).contiguous().view(batch_sz, seq_len, -1)

        return self.rln(out)

class MLP(nn.Module):

    def __init__(self, emb_size: int, scale_ratio: int, act_id: str):
        
        super().__init__()

        self.act    = activators.default_initialize(act_id, round_mlp_emb(emb_size * scale_ratio))
        self.w1     = nn.Linear(emb_size, round_mlp_emb(emb_size * scale_ratio), bias = False)
        self.w2     = RowLinear(round_mlp_emb(emb_size * scale_ratio), emb_size, bias = False)
        self.w3     = nn.Linear(emb_size, round_mlp_emb(emb_size * scale_ratio), bias = False)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:

        return self.w2(self.act(self.w1(emb)) * self.w3(emb))

class Block(nn.Module):

    def __init__(self, emb_size: int, attn_qk_scale_ratio: int, attn_qk_rot_perc: float, attn_v_scale_ratio: int, attn_head_count: int, mlp_scale_ratio: int, act_id: str, lnorm_id: str):

        super().__init__()

        self.attn   = Attention(emb_size, attn_qk_scale_ratio, attn_qk_rot_perc, attn_v_scale_ratio, attn_head_count)
        self.mlp    = MLP(emb_size, mlp_scale_ratio, act_id)
        self.lnorm1 = layernorms.default_initialize(lnorm_id, emb_size)
        self.lnorm2 = layernorms.default_initialize(lnorm_id, emb_size)
    
    def forward(self, emb: torch.Tensor) -> torch.Tensor:

        emb = emb + self.attn(self.lnorm1(emb))
        emb = emb + self.mlp(self.lnorm2(emb))
        
        return emb 

class GPT(nn.Module): 

    def __init__(self, vocab_size: int, emb_size: int, attn_qk_scale_ratio: int, attn_qk_rot_perc: float, attn_v_scale_ratio: int, attn_head_count: int, mlp_scale_ratio: int, block_sz: int, act_id: str, lnorm_id: str):
        
        super().__init__()

        self.emb        = nn.Embedding(vocab_size, emb_size)
        self.blocks     = [Block(emb_size, attn_qk_scale_ratio, attn_qk_rot_perc, attn_v_scale_ratio, attn_head_count, mlp_scale_ratio, act_id, lnorm_id) for _ in range(block_sz)]
        self.ln_out     = nn.Linear(emb_size, vocab_size, bias = False)
        self.lnorm      = layernorms.default_initialize(lnorm_id, emb_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        
        hid = self.emb(sentence)

        for block in self.blocks:
            hid = block(hid)

        return self.ln_out(self.lnorm(hid))