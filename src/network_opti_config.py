training_device                         = ["cuda"]
architecture                            = ["std"]
optimizer                               = ["adamw_no_weight_decay"]
loss_function                           = ["ce"]
lr_scheduler                            = ["cyclic_reduce_on_plateau_lr_5e-5_patience_10"]
mlp_scale_ratio                         = [2/3] #context compression (if mlp scale ratio < 1)
attn_qk_scale_ratio                     = [1, 2, 4]
attn_qk_rot_perc                        = [0.25, 0.5, 0.75]
attn_v_scale_ratio                      = [1, 2, 4]
attn_head_count                         = [8, 16]

activator_id                            = ["serf", "sinsig", "flattenedtswish", "rsigelud", "arelu", "chpaf", "gish", "phish", "eswish", "psgu", "blu", "appsquaredrelu", "swish", "hardswish"]
layernorm_id                            = ["std", "rmsnorm"]
block_sz                                = [4]

max_vocab_sz                            = [1000]
eval_ratio                              = [0.10]
rep_count                               = [3e4]
rep_per_epoch                           = [1e3]
minimum_network_size                    = [1e4] 
maximum_network_size                    = [1e7] 

emb_proj_sz                             = [768]
is_shared_env_ticker_vocab              = [False]

ticker_window_in_year                   = [4]
ticker_resolution                       = [32]
ticker_compression_option               = ["two_level_suffix_encoding"] # + options

ticker_funds_window_in_year             = [8]
ticker_funds_resolution                 = [8]
ticker_funds_compression_option         = ["two_level_suffix_encoding"] # + options

env_window_in_year                      = [8]
env_resolution                          = [4]
env_compression_option                  = ["two_level_suffix_encoding"] # + options

env_has_ep_ratio                        = [False] #important for predicting market crash (earning/price == profitability ratio = eps / low) - tmr work 
env_has_eps                             = [False]
env_has_rev                             = [False]
env_has_inc                             = [False] 
env_has_reddit_sentiment                = [True, False] #need to get done by next week (ticker - sentiment) -> chart -> suffix compress
env_ticker_count                        = [200]
env_option_count                        = [50] 

training_objective                      = ["ticker_sliding_window", "state_sliding_window"] #ticker sliding window's actually a next word predictor, as CE penaltizes wrong suffix prediction heavily (offset)
render_rate_in_day                      = [90]
start_date                              = ["2013-01-01"]
end_date                                = ["2024-01-01"] 
step_in_day                             = [15]
batch_sz                                = [20]