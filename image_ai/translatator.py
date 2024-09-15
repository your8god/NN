from functools import lru_cache

from translate import Translator


@lru_cache
def tr(word):
    t = Translator(from_lang='english', to_lang='russian')
    res = t.translate(word)
    return res
