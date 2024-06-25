import database_api
import utility
import cache 

def extract_yearly_ep_ratio(ticker: str, first_date: str, last_date: str) -> list[float]:
    
    eps_list: list[float]   = database_api.extract_yearly_eps(ticker, first_date, last_date)

    if len(eps_list) == 0:
        return eps_list
    
    daily_list: list[float] = database_api.extract_daily_low_price(ticker, first_date, last_date)
    res: int                = len(eps_list)

    if res < len(daily_list):
        return [] 

    return [eps / price for eps, price in zip(eps_list, utility.low_group(daily_list, res))]

def extract_active_valid_tickers(first_date: str, last_date: str) -> list[str]:

    cache_id: str = "eavt_%s_%s" % (first_date, last_date)
    rs = cache.get_coo_cache(cache_id)

    if rs != None:
        return rs 
    
    tickers: list[str] = database_api.extract_valid_tickers()
    daily_prices: dict[str, list[float]] = database_api.extract_all_daily_low_price(first_date, last_date)
    daily_prices = utility.prune_n_fit(daily_prices, utility.get_expected_trading_days(first_date, last_date))
    rs = list(set(tickers).intersection(set(daily_prices.keys())))
    cache.update_coo_cache(cache_id, rs)
    
    return rs 
