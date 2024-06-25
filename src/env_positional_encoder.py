import taxonomy_id

#consider increasing token vector sz instead of using positional encoding (future consideration)  

def feature_extract(ticker: str) -> list[str]: 

    return [taxonomy_id.sector(ticker), taxonomy_id.market_cap(ticker)]
    
def encode(arr: list[str]) -> list[str]:
    
    scores = [(feature_extract(e), e) for e in arr]
    scores.sort()

    return [e[1] for e in scores]