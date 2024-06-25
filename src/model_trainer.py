from torch import nn 
import torch 
import json 
import os 
import random 

class TrainingConfig:
    model_type: str 
    optimizer: str
    loss_function: str
    lr_scheduler: str
    vocab_sz: int 
    emb_sz: int
    mlp_scale_ratio: int 
    attn_qk_scale_ratio: int 
    attn_qk_rot_perc: float 
    attn_v_scale_ratio: int 
    attn_head_count: int
    activator_id: str
    layernorm_id: str
    block_sz: int

class Config:
    model_config: TrainingConfig
    training_device: str
    reassess_rep: int
    total_rep: int
    eval_ratio: float
    datasource_path: str
    model_output_path: str
    model_report_path: str

def make_config(model_type: str, optimizer: str, loss_function: str, lr_scheduler: str,
                vocab_sz: int, emb_sz: int, mlp_scale_ratio: int, attn_qk_scale_ratio: int,
                attn_qk_rot_perc: float, attn_v_scale_ratio: int, attn_head_count: int,
                activator_id: str, layernorm_id: str, block_sz: int, 
                training_device: str, reassess_rep: int, total_rep: int, eval_ratio: float,
                datasource_path: str, model_output_path: str, model_report_path:str):

    training_config: TrainingConfig = TrainingConfig()
    rs: Config = Config()

    training_config.model_type              = model_type
    training_config.optimizer               = optimizer
    training_config.loss_function           = loss_function
    training_config.lr_scheduler            = lr_scheduler
    training_config.vocab_sz                = vocab_sz
    training_config.emb_sz                  = emb_sz
    training_config.mlp_scale_ratio         = mlp_scale_ratio
    training_config.attn_qk_scale_ratio     = attn_qk_scale_ratio
    training_config.attn_qk_rot_perc        = attn_qk_rot_perc
    training_config.attn_v_scale_ratio      = attn_v_scale_ratio
    training_config.attn_head_count         = attn_head_count
    training_config.activator_id            = activator_id
    training_config.layernorm_id            = layernorm_id
    training_config.block_sz                = block_sz 
    
    rs.model_config                         = training_config
    rs.training_device                      = training_device
    rs.reassess_rep                         = reassess_rep
    rs.total_rep                            = total_rep
    rs.eval_ratio                           = eval_ratio
    rs.datasource_path                      = datasource_path
    rs.model_output_path                    = model_output_path
    rs.model_report_path                    = model_report_path

    return rs 

def parse_model_optimizer_loss_functor_lr_scheduler(training_config: TrainingConfig) ->  tuple[nn.Module, torch.optim.Optimizer, nn.Module, torch.optim.lr_scheduler.LRScheduler]:
    
    model_obj: nn.Module = None  
    optimizer: torch.optim.Optimizer = None 
    loss_functor: nn.Module = None 
    scheduler: torch.optim.lr_scheduler.LRScheduler = None 

    if training_config.model_type.lower() == "llama":
        import model.llama
        raise Exception()
    elif training_config.model_type.lower() == "std":
        import model.std
        model_obj = model.std.GPT(training_config.vocab_sz, training_config.emb_sz, 
                                  training_config.attn_qk_scale_ratio, training_config.attn_qk_rot_perc,
                                  training_config.attn_v_scale_ratio, training_config.attn_head_count,
                                  training_config.mlp_scale_ratio, training_config.block_sz,
                                  training_config.activator_id, training_config.layernorm_id) 

    if training_config.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(params=model_obj.parameters())
    elif training_config.optimizer.lower() == "adamw_no_weight_decay":
        optimizer = torch.optim.AdamW(params=model_obj.parameters(), weight_decay = 0.0)
    else:
        raise Exception()

    if training_config.loss_function.lower() == "ce":
        loss_functor = torch.nn.CrossEntropyLoss()
    else:
        raise Exception()
    
    if training_config.lr_scheduler.lower() == "cyclic_reduce_on_plateau_lr_5e-4_patience_10":
        import lr_scheduler.cyclic_reduce_on_plateau
        scheduler = lr_scheduler.cyclic_reduce_on_plateau.CyclicReduceLROnPlateau(5e-4, 1e-8, optimizer = optimizer, patience = 10)
    elif training_config.lr_scheduler.lower() == "cyclic_reduce_on_plateau_lr_5e-5_patience_10":
        import lr_scheduler.cyclic_reduce_on_plateau
        scheduler = lr_scheduler.cyclic_reduce_on_plateau.CyclicReduceLROnPlateau(5e-5, 1e-8, optimizer = optimizer, patience = 10)
    elif training_config.lr_scheduler.lower() == "cyclic_reduce_on_plateau_lr_5e-6_patience_10":
        import lr_scheduler.cyclic_reduce_on_plateau
        scheduler = lr_scheduler.cyclic_reduce_on_plateau.CyclicReduceLROnPlateau(5e-6, 1e-8, optimizer = optimizer, patience = 10)
    else:
        raise Exception()
    
    return model_obj, optimizer, loss_functor, scheduler

def get_data(datasource_path: str, eval_ratio: float) -> list[tuple[tuple, object]]:
    
    with open(datasource_path, "r") as f:
        arr = json.loads(f.read())

    eval_sz         = int(len(arr) * eval_ratio)
    eval_data       = arr[:eval_sz]
    training_data   = arr[eval_sz:]

    if len(eval_data) == 0 or len(training_data) == 0:
        raise Exception()
     
    return eval_data, training_data

def write_file(p: str, content: str):
    
    with open(p, "w") as f:
        f.write(content)

def make_model_output_path(org_path: str, epoch_idx: int) -> str:

    par_dir     = os.path.dirname(org_path)
    file_name   = os.path.basename(org_path)
    name, ext   = os.path.splitext(file_name)
    new_name    = "%s_%s" % (name, str(epoch_idx)) + ext

    return os.path.join(par_dir, new_name)

def get_validation_loss(net: nn.Module, loss_functor: nn.Module, eval_data: list[tuple[object, object]], training_device: str, sampling_size: int = 30) -> tuple[nn.Module, float]:

    if len(eval_data) == 0:
        raise Exception()
    
    net = net.eval()
    loss_score: float = float(0.0)

    with torch.no_grad():
        for _ in range(sampling_size):
            idx                 = random.randint(0, len(eval_data) - 1)
            in_emb, out_emb     = eval_data[idx]
            in_tensor           = torch.Tensor(in_emb).long().to(training_device)
            out_tensor          = torch.Tensor(out_emb).long().to(training_device)
            net_out             = net(in_tensor).to(training_device)
            total_sentence_sz   = net_out.size()[0] * net_out.size()[1]
            loss                = loss_functor(net_out.view(total_sentence_sz, -1), out_tensor.view(total_sentence_sz))
            loss_score         += loss.item()
        
    return net.train(), loss_score / sampling_size  

def train(config: Config, verbose = True):

    torch.set_default_device(config.training_device)

    net, optim, loss_functor, lr_scheduler = parse_model_optimizer_loss_functor_lr_scheduler(config.model_config)
    rep_per_epoch   = config.reassess_rep
    total_rep       = config.total_rep
    eval_data, data = get_data(config.datasource_path, config.eval_ratio)
    loss_sum        = 0
    min_sum         = 1000000000000
    i               = 0 
    epoch_count     = 0
    report_str      = str()
    net:nn.Module   = net.to(config.training_device)

    torch.set_float32_matmul_precision('high') #
    torch.backends.cudnn.benchmark = True #

    while True: 
        for in_emb, out_emb in data:
            optim.zero_grad()
            in_tensor   = torch.Tensor(in_emb).long().to(config.training_device)
            out_tensor  = torch.Tensor(out_emb).long().to(config.training_device)

            with torch.autocast(device_type=config.training_device, dtype=torch.float16): #
                net_out             = net(in_tensor).to(config.training_device)
                total_sentence_sz   = net_out.size()[0] * net_out.size()[1]
                loss                = loss_functor(net_out.view(total_sentence_sz, -1), out_tensor.view(total_sentence_sz))
                loss.backward()

            loss_sum += loss.item() 
            i += 1
            optim.step()

            if (i % 100 == 0) and verbose:
                print(i)

            if i % rep_per_epoch == 0:
                loss_sum        = loss_sum / rep_per_epoch
                net, eval_loss  = get_validation_loss(net, loss_functor, eval_data, config.training_device)
                min_sum         = min(loss_sum, min_sum)
                report_str      += "training_loss: %s, eval_loss: %s \n" % (str(loss_sum), str(eval_loss))

                if verbose:
                    print("cur_loss: ", loss_sum, "min_loss: ", min_sum, "eval_loss: ", eval_loss)
                
                write_file(config.model_report_path, report_str)
                torch.save(net.state_dict(), make_model_output_path(config.model_output_path, epoch_count))
                lr_scheduler.step(loss_sum)
                loss_sum = 0
                epoch_count += 1

            if i > total_rep:
                if verbose:
                    print("done\n----------------\n")

                return