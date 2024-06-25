import math 
import functools
import cache 

MAX_SUFFIX_BIT_LENGTH   = 32 

def fast_encoding(arr: list[int]) -> int:

    return functools.reduce(lambda lhs, rhs: (lhs << MAX_SUFFIX_BIT_LENGTH) | rhs, arr)

def get_cache(key: list[int]) -> int:

    cache_id: str = "sc_%s" % str(fast_encoding(key))
    return cache.get_cache(cache_id)

def update_cache(key: list[int], val: int):

    cache_id: str = "sc_%s" % str(fast_encoding(key))
    return cache.update_cache(cache_id, val)

def factorial(n: int) -> int:

    return math.factorial(n)

def get_default_suffix_array(n: int) -> list[int]:

    return list(range(n)) 

def is_equal(lhs: list[int], rhs: list[int]) -> bool:

    if len(lhs) != len(rhs):
        return False 

    return not False in [lhs[i] == rhs[i] for i in range(len(lhs))] 
    
def is_suffix_array(arr: list[int]) -> bool:

    dflt: list[int] = get_default_suffix_array(len(arr))
    
    dflt.sort()
    arr = arr.copy()
    arr.sort()
    
    return is_equal(dflt, arr)

def permuted_image_encoding(org: list[int], img: list[int]) -> int:

    assert len(org) == len(img)

    if len(org) == 0:
        return 0
    
    img_idx: int = img.index(org[0])
    val: int = img_idx * factorial(len(org) - 1)
    img.remove(org[0])
        
    return val + permuted_image_encoding(org[1:], img)

def encode(arr: list[int]) -> int:

    # assert is_suffix_array(arr)

    rs = get_cache(arr)

    if rs != None:
        return rs 
    
    rs = permuted_image_encoding(arr, get_default_suffix_array(len(arr)))
    update_cache(arr, rs)

    return rs

def encode_many(*args) -> int:

    return functools.reduce(lambda a, b: encode(a) * factorial(len(b)) + encode(b), args)