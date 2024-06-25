import math 
import database_api
import derivatives_extractor
import utility
from typing import Union

def intersect(lhs: Union[list, set], rhs: Union[list, set]) -> list:

    return list(set(lhs).intersection(set(rhs)))

def dot_product(lhs: list[float], rhs: list[float]) -> float:

    assert len(lhs) == len(rhs) and len(lhs) != 0

    return sum([e * e1 for e, e1 in zip(lhs, rhs)])

def length_to_org(lhs: list[float]) -> float:

    return math.sqrt(dot_product(lhs, lhs)) 

def cos_sim(lhs: list[float], rhs: list[float]) -> float:

    return dot_product(lhs, rhs) / (length_to_org(lhs) * length_to_org(rhs))  

def permuted_randomize_many(arr: list[object], sz: int) -> list[object]:
    
    return utility.shuffle(arr)[:sz]

def get_sector_distribution_vector(arr: list[str]) -> list[int]:
    
    return [len(intersect(arr, database_api.extract_utility_tickers())), len(intersect(arr, database_api.extract_energy_tickers())), len(intersect(arr, database_api.extract_basic_material_tickers())), 
            len(intersect(arr, database_api.extract_industrial_tickers())),len(intersect(arr, database_api.extract_consumer_staple_tickers())), len(intersect(arr, database_api.extract_consumer_discretionary_tickers())), 
            len(intersect(arr, database_api.extract_real_estate_tickers())), len(intersect(arr, database_api.extract_financial_tickers())), len(intersect(arr, database_api.extract_health_care_tickers())), 
            len(intersect(arr, database_api.extract_tele_tickers())), len(intersect(arr, database_api.extract_tech_tickers()))]  

def get_default_sector_distribution_vector() -> list[int]:

    return [len(database_api.extract_utility_tickers()), len(database_api.extract_energy_tickers()), len(database_api.extract_basic_material_tickers()), 
            len(database_api.extract_industrial_tickers()), len(database_api.extract_consumer_staple_tickers()), len(database_api.extract_consumer_discretionary_tickers()), 
            len(database_api.extract_real_estate_tickers()), len(database_api.extract_financial_tickers()), len(database_api.extract_health_care_tickers()), 
            len(database_api.extract_tele_tickers()), len(database_api.extract_tech_tickers())]

def get_market_cap_distribution_vector(arr: list[str]) -> list[int]:
    
    return [len(intersect(arr, database_api.extract_mega_cap_tickers())), len(intersect(arr, database_api.extract_large_cap_tickers())), len(intersect(arr, database_api.extract_mid_cap_tickers()))]

def get_default_market_cap_distribution_vector() -> list[int]:

    return [len(database_api.extract_mega_cap_tickers()), len(database_api.extract_large_cap_tickers()), len(database_api.extract_mid_cap_tickers())]

def get_score(arr: list[str]) -> float:

    return cos_sim(get_sector_distribution_vector(arr) + get_market_cap_distribution_vector(arr), get_default_sector_distribution_vector() + get_default_market_cap_distribution_vector())

def randomize_env_tickers(sz: int, retry_count: int = 1000) -> list[str]:
    
    cand_list: list[list[str]]          = [permuted_randomize_many(derivatives_extractor.extract_active_valid_tickers("2005-01-01", "2024-01-01"), sz) for _ in range(retry_count)] #fix later
    rs: list[tuple[float, list[str]]]   = [(get_score(cand), cand) for cand in cand_list] 
    
    return max(rs)[1]