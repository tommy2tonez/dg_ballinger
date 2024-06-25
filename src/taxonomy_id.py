import database_api

utility                 = set(database_api.extract_utility_tickers())
energy                  = set(database_api.extract_energy_tickers())
basic_material          = set(database_api.extract_basic_material_tickers())
industrial              = set(database_api.extract_industrial_tickers())
consumer_staples        = set(database_api.extract_consumer_staple_tickers())
consumer_discretionary  = set(database_api.extract_consumer_discretionary_tickers())
real_estate             = set(database_api.extract_real_estate_tickers())
financial               = set(database_api.extract_financial_tickers())
health_care             = set(database_api.extract_health_care_tickers())
tele                    = set(database_api.extract_tele_tickers())
tech                    = set(database_api.extract_tech_tickers())

mega_cap                = set(database_api.extract_mega_cap_tickers())
large_cap               = set(database_api.extract_large_cap_tickers())
mid_cap                 = set(database_api.extract_mid_cap_tickers())
small_cap               = set(database_api.extract_small_cap_tickers())

def market_cap(ticker: str) -> str:

    if ticker in mega_cap:
        return "mega"
    
    if ticker in large_cap:
        return "large"
    
    if ticker in mid_cap:
        return "mid"
    
    if ticker in small_cap:
        return "small"
    
    return "unknown"

def sector(ticker: str) -> str:

    if ticker in utility:
        return "u" 
    
    if ticker in energy:
        return "e"

    if ticker in basic_material:
        return "bm"
    
    if ticker in industrial:
        return "i"

    if ticker in consumer_staples:
        return "cs"
    
    if ticker in consumer_discretionary:
        return "cd"
    
    if ticker in real_estate:
        return "re"

    if ticker in financial:
        return "fc"
    
    if ticker in health_care:
        return "hc"
    
    if ticker in tele:
        return "tl"
    
    if ticker in tech:
        return "th"

    return "unknown"