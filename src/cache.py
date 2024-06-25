import json 
import copy 
import utility 

DEFAULT_CACHE_FLUSH_SIZE    = 1e5
DEFAULT_MAX_ENTRY_SIZE      = 1e5
DEFAULT_COPY_ON_OPS         = False 

class Cache:

    def __init__(self, cap: int = DEFAULT_CACHE_FLUSH_SIZE, max_entry_sz: int = DEFAULT_MAX_ENTRY_SIZE, copy_on_ops: bool = DEFAULT_COPY_ON_OPS):

        self.cache          = {}
        self.cap            = cap 
        self.max_entry_sz   = max_entry_sz
        self.copy_on_ops    = copy_on_ops

    def update(self, key: object, value: object):
        
        # if not utility.is_within_byte_size(value, self.max_entry_sz):
        #     return  

        if len(self.cache) >= self.cap:
            self.cache.clear()
 
        if not self.copy_on_ops:
            self.cache[json.dumps(utility.force_json_compatible(key))] = value
        else:
            self.cache[json.dumps(utility.force_json_compatible(key))] = copy.deepcopy(value)

    def get(self, key: object) -> object:

        key_rep = json.dumps(utility.force_json_compatible(key))

        if key_rep not in self.cache:
            return None 
        
        if not self.copy_on_ops:
            return self.cache[key_rep]

        return copy.deepcopy(self.cache[key_rep])
        
global_cache            = Cache()
global_coo_cache        = Cache(copy_on_ops = True)
global_low_cap_cache    = Cache(cap = 1e2)

def get_cache(key: object) -> object:

    global global_cache
    return global_cache.get(key)
 
def update_cache(key: object, value: object):

    global global_cache 
    global_cache.update(key, value)

def get_low_cap_cache(key: object) -> object:

    global global_low_cap_cache
    return global_low_cap_cache.get(key)
 
def update_low_cap_cache(key: object, value: object):

    global global_low_cap_cache 
    global_low_cap_cache.update(key, value)

def get_coo_cache(key: object) -> object:

    global global_coo_cache
    return global_coo_cache.get(key)

def update_coo_cache(key: object, value: object):

    global global_coo_cache
    global_coo_cache.update(key, value)