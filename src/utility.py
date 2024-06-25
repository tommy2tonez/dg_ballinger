import datetime 
import functools 
import random 
import sys 
import json 
from collections.abc import Iterable

class AttributeDict(dict):
  
    def __getattr__(self, name):
        
        return self[name]

#conditionally work
def force_json_compatible(obj: object) -> object:

    if obj == None:
        return obj 
    
    if type(obj) == type(int()):
        return obj

    if type(obj) == type(float()):
        return obj

    if type(obj) == type(str()):
        return obj
    
    if type(obj) == type(bool()):
        return obj
    
    if type(obj) == type(list()):
        return list(map(force_json_compatible, obj)) 
    
    if type(obj) == type(dict()):
        return dict(force_json_compatible(list(obj.items())))

    if type(obj) == type(tuple()):
        return tuple(force_json_compatible(list(obj)))
    
    if type(obj) == type(set()):
        return force_json_compatible(list(obj))
    
    return force_json_compatible(obj.__dict__)  #bad assumption - should be in precond

#conditionally work
def force_object_compatible(obj: object) -> object:
    
    if obj == None:
        return obj 
    
    if type(obj) == type(int()):
        return obj 
    
    if type(obj) == type(float()):
        return obj 
    
    if type(obj) == type(str()):
        return obj 
    
    if type(obj) == type(bool()):
        return obj 
    
    if type(obj) == type(list()):
        return list(map(force_object_compatible, obj))

    if type(obj) == type(dict()):
        return AttributeDict(dict(force_object_compatible(list(obj.items()))))

    if type(obj) == type(tuple()):
        return tuple(force_object_compatible(list(obj)))
    
    if type(obj) == type(set()):
        return set(force_object_compatible(list(obj)))
    
    return obj #bad assumption - should be in precond

def dump_to_file(out_path: str, *args, **kwargs):

    rs = {"args": args, "kwargs": kwargs}

    with open(out_path, "w") as f:
        f.write(json.dumps(force_json_compatible(rs)))

def byte_size(obj: object) -> int:

    if obj == None or type(obj) == type(int()) or type(obj) == type(bool()) or type(obj) == type(float()) or type(obj) == type(str()) or type(obj) == type(complex()):
        return sys.getsizeof(obj)

    if type(obj) == type(list()) or type(obj) == type(set()) or type(obj) == type(tuple()):
        return sum(map(byte_size, obj)) + sys.getsizeof(obj)  
    
    if type(obj) == type(dict()):
        return sum(map(byte_size, obj.items())) + sys.getsizeof(obj) 
    
    raise Exception()

def array_split(inp: list[object], resolution: int) -> list[list[object]]:

    assert len(inp) >= resolution

    step    = len(inp) // resolution
    rs      = []

    for i in range(resolution):
        f = i * step 
        l = f + step if i != resolution - 1 else len(inp)
        rs += [inp[f:l]]

    return rs

def low_group(inp: list[object], resolution: int) -> list[object]:
    
    return list(map(min, array_split(inp, resolution)))

def right_window(arr: list[object], sz: int) -> list[object]:

    if sz > len(arr):
        return [] 
    
    return arr[len(arr) - sz:]

def year_to_day(yrs: int) -> int:

    return yrs * 365

def month_to_day(months: int) -> int:

    return months * 30 

def day_to_trading_day(days: int) -> int:

    return int(days * float(4.5) / 7) 

def trading_day_to_day(days: int) -> int:

    return int(days * float(7) / 4.5)

def date_obj_to_std_date_str(obj: datetime.datetime) -> str:

    return datetime.datetime.strftime(obj, "%Y-%m-%d") 

def std_date_str_to_date_obj(obj: str) -> datetime.datetime:
    
    return datetime.datetime.strptime(obj, "%Y-%m-%d")

def get_posix_date() -> datetime.datetime:

    return datetime.datetime(1970,1,1)

def date_obj_to_delta_posix_day(d: datetime.datetime) -> int:

    return (d - get_posix_date()).days

def delta_posix_day_to_date_obj(day_count: int) -> datetime.datetime:

    return get_posix_date() + datetime.timedelta(days = day_count)

def get_expected_trading_days(first_date: str, last_date: str) -> int: 
    
    first_date_obj  = std_date_str_to_date_obj(first_date)
    last_date_obj   = std_date_str_to_date_obj(last_date)
    delta_days      = (last_date_obj - first_date_obj).days

    return day_to_trading_day(delta_days)

def shuffle(arr: list[object], iter_mul_factor: int = 10) -> list:

    for _ in range(len(arr) * iter_mul_factor):
        lhs = random.randint(0, len(arr) - 1)
        rhs = random.randint(0, len(arr) - 1)
        tmp = arr[lhs]
        arr[lhs] = arr[rhs]
        arr[rhs] = tmp 
    
    return arr

def pair_batchify(arr: list[tuple[object, object]], batch_sz: int) -> list[tuple[list[object], list[object]]]:
    
    num_batch: int = len(arr) // batch_sz
    rs = []

    for i in range(num_batch):
        b           = i * batch_sz
        e           = (i + 1) * batch_sz
        in_batch    = [arr[j][0] for j in range(b,e)]
        out_batch   = [arr[j][1] for j in range(b,e)]
        rs          += [(in_batch, out_batch)]
    
    return rs

def array_transform_to_0i_sum(arr: list[int]) -> list[int]:

    return [sum(arr[0:i]) for i in range(len(arr))]

def vocab_join(lhs: list[int], rhs: list[int]) -> list[int]:

    if len(lhs) == 0:
        return lhs + rhs 
    
    lhs_vocab_sz = max(lhs)
    rhs = [e + lhs_vocab_sz for e in rhs]

    return lhs + rhs
