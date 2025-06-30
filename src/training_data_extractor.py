import json 
import datetime 
import utility
import time_series_compression
import database_api
import cache
from typing import Optional

#we'd want to improve this

#let's see the 30 strings that made up the stock movement:

#the previous movements (the ticker's history) string
#trait string
#nationality string
#sector string (DFEN | NRGU)
#market string (SPY)
#mood string
#fundamental string
#peer (entity resolution) string (empathy string) by doing graph search of connected entities (we extract the maybe related entities as env_ticker_list: list[list[str]] and feed to the neural network)
#random string
#rational string
#catalyst string
#illegal string (pump and dump)
#unprecended string (hype string)
#option string (available contract)
#window of validation string, new trends in 2024 have to be different than that of 1993
#the conservation of money string (this is probably very important), and their growth

#let's see what we could do
#traits extraction by using focal points, problem, we'd want to move the focal to exactly to where the trait repeats
#without loss of generality, we'd want to uniformly split a time window [a, b], for exponentially increasing b
#calls/ puts chart (this is hard to collect)
#rational decisions (cant afford to go off a cliff, repeat history, has to have some Big News, too many people on the line) 
#illegal activities (pump and dump, money laundering, etc.)

#empathy string establishment by using graph hops (entity resolution) for env tickers
#we'd want to include that in the training data input and output

#Fear/ Greed/ Mood by doing sampling of news

#Fundamental strings, essentially, we'd want to include/ compress every fundamental chart inside the input/output result

class StateEmbeddingConfig:
    is_shared_env_ticker_vocab: bool 

    ticker_window_in_year: int 
    ticker_resolution: int
    ticker_compression_option: str 

    ticker_fundamental_window_in_year: int
    ticker_fundamental_resolution: int 
    ticker_fundamental_compression_option: str

    env_window_in_year: int 
    env_resolution: int 
    env_compression_option: str
    env_has_eps: bool 
    env_has_rev: bool 
    env_has_inc: bool
    env_ticker_list: list[list[str]]

class Config: 
    state_emb_config: StateEmbeddingConfig
    start_date: str 
    end_date: str
    render_rate_in_day: int
    delta_step_in_day: int 
    batch_sz: int
    output_file: str 

def make_config(is_shared_env_ticker_vocab: bool, ticker_window_in_year: int, ticker_resolution: int, ticker_compression_option: str, 
                ticker_fundamental_window_in_year: int, ticker_fundamental_resolution: int, ticker_fundamental_compression_option: str, 
                env_window_in_year: int, env_resolution: int, env_compression_option: str, 
                env_has_eps: bool, env_has_rev: bool, env_has_inc: bool, env_ticker_list: list[list[str]],
                render_rate_in_day: int,
                start_date: str, end_date: str, delta_step_in_day: int, batch_sz: int, output_file: str) -> Config:
    
    state_emb_config: StateEmbeddingConfig                  = StateEmbeddingConfig()
    rs: Config                                              = Config() 

    state_emb_config.is_shared_env_ticker_vocab             = is_shared_env_ticker_vocab 

    state_emb_config.ticker_window_in_year                  = ticker_window_in_year
    state_emb_config.ticker_resolution                      = ticker_resolution
    state_emb_config.ticker_compression_option              = ticker_compression_option 

    state_emb_config.ticker_fundamental_window_in_year      = ticker_fundamental_window_in_year
    state_emb_config.ticker_fundamental_resolution          = ticker_fundamental_resolution
    state_emb_config.ticker_fundamental_compression_option  = ticker_fundamental_compression_option 

    state_emb_config.env_window_in_year                     = env_window_in_year
    state_emb_config.env_resolution                         = env_resolution
    state_emb_config.env_compression_option                 = env_compression_option
    state_emb_config.env_has_eps                            = env_has_eps
    state_emb_config.env_has_rev                            = env_has_rev
    state_emb_config.env_has_inc                            = env_has_inc
    state_emb_config.env_ticker_list                        = env_ticker_list

    rs.state_emb_config                                     = state_emb_config
    rs.render_rate_in_day                                   = render_rate_in_day
    rs.start_date                                           = start_date
    rs.end_date                                             = end_date
    rs.delta_step_in_day                                    = delta_step_in_day
    rs.batch_sz                                             = batch_sz
    rs.output_file                                          = output_file

    return rs

def make_ticker_state(ticker: str, state_emb_config: StateEmbeddingConfig, to_date: str) -> Optional[list]:

    ticker_fr_date: str                 = utility.date_obj_to_std_date_str(utility.std_date_str_to_date_obj(to_date) - datetime.timedelta(days = utility.year_to_day(state_emb_config.ticker_window_in_year)))
    fundamental_fr_date: str            = utility.date_obj_to_std_date_str(utility.std_date_str_to_date_obj(to_date) - datetime.timedelta(days = utility.year_to_day(state_emb_config.ticker_fundamental_window_in_year + 1)))
    ticker_price_list: list[float]      = database_api.extract_daily_low_price(ticker, ticker_fr_date, to_date)
    income_price_list: list[float]      = utility.right_window(database_api.extract_yearly_income(ticker, fundamental_fr_date, to_date), state_emb_config.ticker_fundamental_window_in_year)
    revenue_price_list: list[float]     = utility.right_window(database_api.extract_yearly_revenue(ticker, fundamental_fr_date, to_date), state_emb_config.ticker_fundamental_window_in_year)
    eps_price_list: list[float]         = utility.right_window(database_api.extract_yearly_eps(ticker, fundamental_fr_date, to_date), state_emb_config.ticker_fundamental_window_in_year)
    ticker_embedding_list: list[int]    = [] 
    income_embedding_list: list[int]    = []
    revenue_embedding_list: list[int]   = []
    eps_embedding_list: list[int]       = []

    if len(ticker_price_list) < state_emb_config.ticker_resolution or len(ticker_price_list) < utility.get_expected_trading_days(ticker_fr_date, to_date):
        return None 
    
    if len(income_price_list) < state_emb_config.ticker_fundamental_window_in_year or len(income_price_list) < state_emb_config.ticker_fundamental_resolution:
        return None 
    
    if len(revenue_price_list) < state_emb_config.ticker_fundamental_window_in_year or len(income_price_list) < state_emb_config.ticker_fundamental_resolution:
        return None 
    
    if len(eps_price_list) < state_emb_config.ticker_fundamental_window_in_year or len(income_price_list) < state_emb_config.ticker_fundamental_resolution:
        return None 
    
    if state_emb_config.ticker_compression_option == "two_level_suffix_encoding":
        ticker_embedding_list   = time_series_compression.encode_to_suffix_array(ticker_price_list, state_emb_config.ticker_resolution, 2, True)
    else:
        return None 
    
    if state_emb_config.ticker_fundamental_compression_option == "two_level_suffix_encoding":
        income_embedding_list   = time_series_compression.encode_to_suffix_array(income_price_list, state_emb_config.ticker_fundamental_resolution, 2, True)
        revenue_embedding_list  = time_series_compression.encode_to_suffix_array(revenue_price_list, state_emb_config.ticker_fundamental_resolution, 2, True)
        eps_embedding_list      = time_series_compression.encode_to_suffix_array(eps_price_list, state_emb_config.ticker_fundamental_resolution, 2, True)
    else:
        return None 
    
    return ticker_embedding_list + income_embedding_list + revenue_embedding_list + eps_embedding_list

def make_env_state(state_emb_config: StateEmbeddingConfig, to_date: str) -> list:

    fr_date: str                = utility.date_obj_to_std_date_str(utility.std_date_str_to_date_obj(to_date) - datetime.timedelta(days = utility.year_to_day(state_emb_config.env_window_in_year)))
    funds_fr_date: str          = utility.date_obj_to_std_date_str(utility.std_date_str_to_date_obj(to_date) - datetime.timedelta(days = utility.year_to_day(state_emb_config.env_window_in_year + 1)))
    env_ticker_list: list[str]  = state_emb_config.env_ticker_list[0]
    cache_id: str               = json.dumps(["make_env_state", fr_date, to_date, env_ticker_list])
    rs                          = cache.get_cache(cache_id)

    if rs != None: 
        return rs 

    offset_arr                  = [1, int(state_emb_config.env_has_eps), int(state_emb_config.env_has_rev), int(state_emb_config.env_has_inc)]
    feature_sz: int             = sum(offset_arr)
    offset_arr                  = utility.array_transform_to_0i_sum(offset_arr)
    bucket_sz: int              = feature_sz * len(env_ticker_list)
    rs                          = [0] * bucket_sz

    for idx, ticker in enumerate(env_ticker_list):
        price_list: list[float] = database_api.extract_daily_low_price(ticker, fr_date, to_date)

        if len(price_list) >= state_emb_config.env_resolution and len(price_list) >= utility.get_expected_trading_days(fr_date, to_date) and  state_emb_config.env_compression_option == "two_level_suffix_encoding":
            bucket_idx      = idx * feature_sz + offset_arr[0]
            rs[bucket_idx]  = time_series_compression.encode_to_int(price_list, state_emb_config.env_resolution, 2)
    
    for idx, ticker in enumerate(env_ticker_list):
        eps_list: list[float]   = utility.right_window(database_api.extract_yearly_eps(ticker, funds_fr_date, to_date), state_emb_config.env_window_in_year)
        rev_list: list[float]   = utility.right_window(database_api.extract_yearly_revenue(ticker, funds_fr_date, to_date), state_emb_config.env_window_in_year)
        inc_list: list[float]   = utility.right_window(database_api.extract_yearly_income(ticker, funds_fr_date, to_date), state_emb_config.env_window_in_year)

        if len(eps_list) >= state_emb_config.env_window_in_year and len(eps_list) >= state_emb_config.env_resolution and state_emb_config.env_has_eps and state_emb_config.env_compression_option == "two_level_suffix_encoding":
            bucket_idx      = idx * feature_sz + offset_arr[1]
            rs[bucket_idx]  = time_series_compression.encode_to_int(eps_list, state_emb_config.env_resolution, 2)
    
        if len(rev_list) >= state_emb_config.env_window_in_year and len(rev_list) >= state_emb_config.env_resolution and state_emb_config.env_has_rev and state_emb_config.env_compression_option == "two_level_suffix_encoding":
            bucket_idx      = idx * feature_sz + offset_arr[2]
            rs[bucket_idx]  = time_series_compression.encode_to_int(rev_list, state_emb_config.env_resolution, 2)

        if len(inc_list) >= state_emb_config.env_window_in_year and len(inc_list) >= state_emb_config.env_resolution and state_emb_config.env_has_inc and state_emb_config.env_compression_option == "two_level_suffix_encoding":
            bucket_idx      = idx * feature_sz + offset_arr[3]
            rs[bucket_idx]  = time_series_compression.encode_to_int(inc_list, state_emb_config.env_resolution, 2)
    
    cache.update_cache(cache_id, rs)

    return rs

def make_state(ticker: str, state_emb_config: StateEmbeddingConfig, to_date: str) -> list:
    
    ticker_state    = make_ticker_state(ticker, state_emb_config, to_date)
    env_state       = make_env_state(state_emb_config, to_date) 

    if ticker_state == None:
        return None 
    
    if not state_emb_config.is_shared_env_ticker_vocab: 
        return utility.vocab_join(ticker_state, env_state)

    return ticker_state + env_state

def extract(config: Config, verbose = True):
    
    valid_tickers   = database_api.extract_valid_tickers()
    first_date_obj  = utility.std_date_str_to_date_obj(config.start_date)
    last_date_obj   = utility.std_date_str_to_date_obj(config.end_date)
    day_count       = (last_date_obj - first_date_obj).days - config.render_rate_in_day
    rs: list        = list() 

    for i in range(0, day_count, config.delta_step_in_day):
        to_date         = first_date_obj + datetime.timedelta(days = i)
        future_to_date  = to_date + datetime.timedelta(days = config.render_rate_in_day)

        if verbose:
            print(utility.date_obj_to_std_date_str(to_date))

        for ticker in valid_tickers:
            utility.shuffle(config.state_emb_config.env_ticker_list)
            cur_state       = make_state(ticker, config.state_emb_config, utility.date_obj_to_std_date_str(to_date))
            future_state    = make_state(ticker, config.state_emb_config, utility.date_obj_to_std_date_str(future_to_date))

            if cur_state != None and future_state != None:
                rs += [(cur_state, future_state)]

    if verbose:
        print("shuffling, length = ", len(rs))

    rs = utility.pair_batchify(utility.shuffle(rs), config.batch_sz) 

    with open(config.output_file, "w") as f:
        f.write(json.dumps(rs))

    if verbose:
        print("done\n--------------------\n")