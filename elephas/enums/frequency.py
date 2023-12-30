import sys

if sys.version_info.minor < 11:
    # Devs using version < 3.11 can use the str enum mixin
    from enum import Enum


    class Frequency(str, Enum):
        EPOCH: str = 'epoch'
        BATCH: str = 'batch'
else:
    # Devs using version >= 3.11 can use the strenum builtin,
    # but that breaks the mixin implementation
    # https://github.com/python/cpython/issues/100458
    from enum import StrEnum, auto


    class Frequency(StrEnum):
        EPOCH = auto()
        BATCH = auto()
