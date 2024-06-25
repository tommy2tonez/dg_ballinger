import network_opti_config
import model_trainer
import training_data_extractor
import random 
import env_fair_randomizer
import env_positional_encoder

def randomize(arr: list[object]) -> object:

    if len(arr) == 0:
        return None 
    
    return arr[random.randint(0, len(arr) - 1)] 
    
def net_size(emb_proj_sz: int, attn_scale_ratio: int, mlp_scale_ratio: int, block_sz: int) -> int:
    
    return (emb_proj_sz * attn_scale_ratio * 3 + (emb_proj_sz * mlp_scale_ratio * 2)) * block_sz * 2

def randomize_training_config(model_output_path: str, model_report_path: str, training_data_path: str) -> model_trainer.Config:

    training_device: str        = randomize(network_opti_config.training_device)
    architecture:str            = randomize(network_opti_config.architecture)
    optimizer:str               = randomize(network_opti_config.optimizer)
    loss_function: str          = randomize(network_opti_config.loss_function)
    lr_scheduler: str           = randomize(network_opti_config.lr_scheduler)
    mlp_scale_ratio: int        = randomize(network_opti_config.mlp_scale_ratio)
    attn_qk_scale_ratio: int    = randomize(network_opti_config.attn_qk_scale_ratio)
    attn_qk_rot_perc: float     = randomize(network_opti_config.attn_qk_rot_perc)
    attn_v_scale_ratio: int     = randomize(network_opti_config.attn_v_scale_ratio)
    attn_head_count: int        = randomize(network_opti_config.attn_head_count)
    act_id: str                 = randomize(network_opti_config.activator_id)
    lnorm_id: str               = randomize(network_opti_config.layernorm_id)
    block_sz: int               = randomize(network_opti_config.block_sz)
    max_vocab_sz: int           = randomize(network_opti_config.max_vocab_sz)
    eval_ratio: float           = randomize(network_opti_config.eval_ratio)
    rep_count: int              = randomize(network_opti_config.rep_count)
    rep_per_epoch: int          = randomize(network_opti_config.rep_per_epoch)
    min_nw_sz: int              = randomize(network_opti_config.minimum_network_size)
    max_nw_sz: int              = randomize(network_opti_config.maximum_network_size)
    emb_proj_sz: int            = randomize(network_opti_config.emb_proj_sz)
    net_sz: int                 = net_size(emb_proj_sz, max(attn_qk_scale_ratio, attn_v_scale_ratio), mlp_scale_ratio, block_sz)
    
    if net_sz < min_nw_sz or net_sz > max_nw_sz:
        return randomize_training_config(model_output_path, model_report_path, training_data_path)
    
    trainer_config   = model_trainer.make_config(architecture, optimizer, loss_function, lr_scheduler, max_vocab_sz, emb_proj_sz, mlp_scale_ratio, 
                                                 attn_qk_scale_ratio, attn_qk_rot_perc, attn_v_scale_ratio, attn_head_count, act_id, lnorm_id, 
                                                 block_sz, training_device, rep_per_epoch, rep_count, eval_ratio, training_data_path, 
                                                 model_output_path, model_report_path)
    
    return trainer_config

def randomize_extractor_config(training_data_path: str) -> training_data_extractor.Config:
    
    is_shared_env_ticker_vocab: bool        = randomize(network_opti_config.is_shared_env_ticker_vocab)
    ticker_window_in_year: int              = randomize(network_opti_config.ticker_window_in_year)
    ticker_resolution: int                  = randomize(network_opti_config.ticker_resolution)
    ticker_compression_option: str          = randomize(network_opti_config.ticker_compression_option)

    ticker_funds_window_in_year: int        = randomize(network_opti_config.ticker_funds_window_in_year)
    ticker_funds_resolution: int            = randomize(network_opti_config.ticker_funds_resolution)
    ticker_funds_compression_option: str    = randomize(network_opti_config.ticker_funds_compression_option)

    env_window_in_year: int                 = randomize(network_opti_config.env_window_in_year)
    env_compression_option: str             = randomize(network_opti_config.env_compression_option)
    env_resolution: int                     = randomize(network_opti_config.env_resolution)
    env_has_eps: bool                       = randomize(network_opti_config.env_has_eps)
    env_has_rev: bool                       = randomize(network_opti_config.env_has_rev)
    env_has_inc: bool                       = randomize(network_opti_config.env_has_inc)
    env_ticker_count: int                   = randomize(network_opti_config.env_ticker_count)
    env_option_count: int                   = randomize(network_opti_config.env_option_count)
    env_ticker_list: list[list[str]]        = [env_positional_encoder.encode(env_fair_randomizer.randomize_env_tickers(env_ticker_count)) for _ in range(env_option_count)]

    render_rate_in_day: int                 = randomize(network_opti_config.render_rate_in_day)
    start_date: str                         = randomize(network_opti_config.start_date)
    end_date: str                           = randomize(network_opti_config.end_date)
    step_in_day: int                        = randomize(network_opti_config.step_in_day)
    batch_sz: int                           = randomize(network_opti_config.batch_sz)

    
    extractor_config = training_data_extractor.make_config(is_shared_env_ticker_vocab, ticker_window_in_year, ticker_resolution, ticker_compression_option,
                                                           ticker_funds_window_in_year, ticker_funds_resolution, ticker_funds_compression_option,
                                                           env_window_in_year, env_resolution, env_compression_option,
                                                           env_has_eps, env_has_rev, env_has_inc, env_ticker_list,
                                                           render_rate_in_day,
                                                           start_date, end_date, step_in_day, batch_sz, training_data_path)

    return extractor_config
