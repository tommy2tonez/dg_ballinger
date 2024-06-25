import glob 
import os 
import json 
import cache 
import datetime 
import data 
import itertools
import data.taxonomy
import data.taxonomy.market_cap_taxonomy
import data.taxonomy.sector_taxonomy  
import copy 
import utility

DATABASE_PATH       = os.path.dirname(data.__file__)
FUNDAMENTAL_PATH    = os.path.join(DATABASE_PATH, "income_sheets")
TICKER_PATH         = os.path.join(DATABASE_PATH, "gg_daily_parsed")

def cached_read_json_file(fp: str) -> object:

    cache_id = "crjf_%s" % fp
    rs = cache.get_cache(cache_id)

    if rs != None:
        return rs  

    with open(fp, "r") as f_in:
        rs = json.loads(f_in.read())
    
    cache.update_cache(cache_id, rs)
    return rs

def cached_glob(query: str) -> object:

    cache_id = "cg_%s" % query
    rs = cache.get_cache(cache_id)

    if rs != None:
        return rs  

    rs = glob.glob(query) 
    cache.update_cache(cache_id, rs)

    return rs  

def extract_mega_cap_tickers() -> list[str]:
    
    return list(data.taxonomy.market_cap_taxonomy.mega)

def extract_large_cap_tickers() -> list[str]:
    
    return list(data.taxonomy.market_cap_taxonomy.large)

def extract_mid_cap_tickers() -> list[str]:
    
    return list(data.taxonomy.market_cap_taxonomy.mid)

def extract_small_cap_tickers() -> list[str]:
    
    return list(data.taxonomy.market_cap_taxonomy.small)

def extract_utility_tickers() -> list[str]:

    return list(data.taxonomy.sector_taxonomy.utility) 

def extract_energy_tickers() -> list[str]:

    return list(data.taxonomy.sector_taxonomy.energy)

def extract_basic_material_tickers() -> list[str]:

    return list(data.taxonomy.sector_taxonomy.basic_material)

def extract_industrial_tickers() -> list[str]:

    return list(data.taxonomy.sector_taxonomy.industrial)

def extract_consumer_staple_tickers() -> list[str]:

    return list(data.taxonomy.sector_taxonomy.consumer_staples)

def extract_consumer_discretionary_tickers() -> list[str]:

    return list(data.taxonomy.sector_taxonomy.consumer_discretionary)

def extract_real_estate_tickers() -> list[str]:

    return list(data.taxonomy.sector_taxonomy.real_estate)

def extract_financial_tickers() -> list[str]:

    return list(data.taxonomy.sector_taxonomy.financial)

def extract_health_care_tickers() -> list[str]:

    return list(data.taxonomy.sector_taxonomy.health_care)

def extract_tele_tickers() -> list[str]:

    return list(data.taxonomy.sector_taxonomy.tele)

def extract_tech_tickers() -> list[str]:

    return list(data.taxonomy.sector_taxonomy.tech)

def extract_valid_tickers() -> list[str]:
    
    paths = glob.glob(os.path.join(FUNDAMENTAL_PATH, "*.json")) 
    return list(set([os.path.splitext(os.path.basename(path))[0] for path in paths]))

def extract_fundamental(ticker: str, first_date: str, last_date: str, tag: str) -> list[float]:
    
    first_year  = datetime.datetime.strptime(first_date, "%Y-%m-%d").year
    last_year   = datetime.datetime.strptime(last_date, "%Y-%m-%d").year
    path        = os.path.join(FUNDAMENTAL_PATH, "%s.json" % ticker)
    content     = cached_read_json_file(path)
    rs          = []

    for e in content["report"]:
        fiscal_year = int(e["calendarYear"])
        if first_year <= fiscal_year and fiscal_year < last_year:
            rs += [float(e[tag])]
    
    return rs[::-1]

def extract_yearly_revenue(ticker: str, first_date: str, last_date: str) -> list[float]:
    
    return extract_fundamental(ticker, first_date, last_date, "revenue")

def extract_yearly_income(ticker: str, first_date: str, last_date: str) -> list[float]:
    
    return extract_fundamental(ticker, first_date, last_date, "netIncome")

def extract_yearly_eps(ticker: str, first_date: str, last_date: str) -> list[float]:

    return extract_fundamental(ticker, first_date, last_date, "eps")

def extract_yearly_income_ratio(ticker: str, first_date: str, last_date: str) -> list[float]:
    
    return extract_fundamental(ticker, first_date, last_date, "netIncomeRatio")

def extract_all_daily_low_price_base(first_date: str, last_date: str) -> dict[str, list[float]]:

    files: list[str] = cached_glob(os.path.join(TICKER_PATH, "*"))
    price_list: dict[str, list[int]] = {} 
    files.sort()

    for file in files:
        
        if os.path.basename(file) < first_date or os.path.basename(file) >= last_date:
            continue 
        
        data = cached_read_json_file(file)

        for ticker in data:
            if ticker["T"] not in price_list:
                price_list[ticker["T"]] = []

            if "l" in ticker:
                price_list[ticker["T"]] += [ticker["l"]]
    
    return price_list

def extract_all_daily_low_price_recursive(delta_posix_day_first: int, delta_posix_day_last: int, delta_posix_interval_day_first: int, delta_posix_interval_day_last: int) -> dict[str, object]: # return iterable (list or iterator)
    
    rs = {}

    if delta_posix_day_first == delta_posix_interval_day_first and delta_posix_day_last == delta_posix_interval_day_last:
        cache_id = "adlpr_%s_%s" % (str(delta_posix_day_first), str(delta_posix_day_last))
        rs = cache.get_cache(cache_id)
        if rs != None:
            return rs
        rs = extract_all_daily_low_price_base(utility.date_obj_to_std_date_str(utility.delta_posix_day_to_date_obj(delta_posix_day_first)), 
                                              utility.date_obj_to_std_date_str(utility.delta_posix_day_to_date_obj(delta_posix_day_last)))
        cache.update_cache(cache_id, rs)
        return rs 
    
    mid: int = (delta_posix_interval_day_first + delta_posix_interval_day_last) // 2

    if delta_posix_day_last <= mid:
        return extract_all_daily_low_price_recursive(delta_posix_day_first, delta_posix_day_last, delta_posix_interval_day_first, mid)
    
    if delta_posix_day_first >= mid: 
        return extract_all_daily_low_price_recursive(delta_posix_day_first, delta_posix_day_last, mid, delta_posix_interval_day_last)
    
    lhs: dict[str, object]  = extract_all_daily_low_price_recursive(delta_posix_day_first, mid, delta_posix_interval_day_first, mid)
    rhs: dict[str, object]  = extract_all_daily_low_price_recursive(mid, delta_posix_day_last, mid, delta_posix_interval_day_last)
    all_keys: set[str]      = set(lhs.keys()).union(set(rhs.keys()))

    for key in all_keys:
        if key in lhs and key in rhs:
            rs[key] = itertools.chain(lhs[key], rhs[key])
        elif key in lhs:
            rs[key] = lhs[key]
        else:
            rs[key] = rhs[key]

    return rs

#return an immutable object (for performance) - UB if mutate 
def extract_all_daily_low_price(first_date: str, last_date: str) -> dict[str, list[float]]:

    cache_id    = "adlp_%s_%s" % (first_date, last_date)
    rs          = cache.get_low_cap_cache(cache_id)

    if rs != None:
        return rs

    interval_first: int = 0
    interval_last: int  = 1 << 15
    arg_first: int      = utility.date_obj_to_delta_posix_day(utility.std_date_str_to_date_obj(first_date))
    arg_last: int       = utility.date_obj_to_delta_posix_day(utility.std_date_str_to_date_obj(last_date))

    if arg_first >= arg_last:
        return {}
    
    rs  = extract_all_daily_low_price_recursive(arg_first, arg_last, interval_first, interval_last)
    rs  = {ticker: list(rs[ticker]) for ticker in rs}
    cache.update_low_cap_cache(cache_id, rs)

    return rs

def extract_daily_low_price(ticker: str, first_date: str, last_date: str) -> list[float]:

    all_daily = extract_all_daily_low_price(first_date, last_date)
    return copy.deepcopy(all_daily[ticker]) if ticker in all_daily else []