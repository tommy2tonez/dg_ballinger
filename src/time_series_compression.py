import suffix_compression
import utility

def adjecent_perc_diff(arr: list[float], epsilon: float) -> list[float]:

    if len(arr) == 0:
        return arr

    return [arr[0]] + [arr[i] / max(arr[i - 1], epsilon) for i in range(1, len(arr))]

def lossless_compress(inp: list[float], epsilon: float = 0.001) -> tuple[list[int], list[float]]:
    
    inp = [(element, idx) for idx, element in enumerate(inp)]
    inp.sort()

    suffix_arr  = [idx for _, idx in inp] 
    element_arr = [e for e, _ in inp]

    return suffix_arr, adjecent_perc_diff(element_arr, epsilon)

def encode_to_suffix_array(arr: list[float], res: int, recursive_size: int, is_shared_vocab: bool = True) -> list[int]:
    
    if recursive_size == 0:
        return []
    
    suffix_arr, perc_arr    = lossless_compress(utility.low_group(arr, res))
    nxt_suffix_arr          = encode_to_suffix_array(perc_arr, res, recursive_size - 1, is_shared_vocab)

    if is_shared_vocab:
        return suffix_arr + nxt_suffix_arr
    
    return utility.vocab_join(suffix_arr, nxt_suffix_arr)

def encode_to_int(arr: list[float], res: int, recursive_size: int) -> int:

    encoded: list[int]          = encode_to_suffix_array(arr, res, recursive_size, True)
    spllited: list[list[int]]   = utility.array_split(encoded, recursive_size)
    
    return suffix_compression.encode_many(*spllited) 