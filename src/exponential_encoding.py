def clamp(tgt: object, low: object, high: object) -> object:

    if tgt < low:
        return low 

    if tgt > high: 
        return high 

    return tgt 
 
def get_exponent(tgt: float, e0: float, exp_base: float):

    LARGE_VALUE = 1000

    for i in range(LARGE_VALUE):
        if tgt < e0 * (exp_base ** i):
            return i 

    raise Exception() 

def get_e0(span: float, res: int, exp_base: float):
    
    assert res > 0 and exp_base > 0
    
    return span / (exp_base ** (res - 1))

def exponential_encoding(arr: list[float], low: float, high: float, resolution: int, exp_base: float) -> list[int]:

    #span = e0 * exp_base^res 
    #e0 = span/exp_base**res
    
    assert resolution > 0 and exp_base > 0

    arr = [clamp(e, low, high) for e in arr]
    arr = [e - low for e in arr] 
    e0  = get_e0(high - low, resolution, exp_base) 

    return [min(get_exponent(e, e0, exp_base), resolution - 1) for e in arr]

def signed_exponential_encoding(arr: list[float], low: float, high: float, resolution: int, exp_base: float) -> list[int]:
    
    assert high > low 

    if low > 0:
        return exponential_encoding(arr, low, high, resolution, exp_base)

    low_res = int((abs(low) / (high - low)) * resolution)
    hi_res  = resolution - low_res 
    rs      = []

    for e in arr:
        appender = []

        if e < 0:
            appender = exponential_encoding([e], 0, abs(low), low_res, exp_base)
        else:
            appender = exponential_encoding([e], 0, high, hi_res, exp_base)
            appender[0] += low_res 
        
        rs += appender
    
    return rs 

def initial_encoding(arr: list[float]) -> list[float]:

    if len(arr) == 0:
        return []

    if arr[0] > 0:
        return [1] + arr[1:]

    if arr[0] < 0:
        return [-1] + arr[1:]

    return arr 