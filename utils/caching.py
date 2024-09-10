import functools
from cachetools import TTLCache
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

cache_expiration = config['cache']['expiration']

cache = TTLCache(maxsize=100, ttl=cache_expiration)

def cache_result(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key in cache:
            return cache[key]
        result = await func(*args, **kwargs)
        cache[key] = result
        return result
    return wrapper



