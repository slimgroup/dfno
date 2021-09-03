from typing import Any, List, Iterator, TypeVar

T = TypeVar('T')

def window_iter(l: List[T], window: int) -> Iterator[List[T]]:

    assert len(l) >= window

    for i in range(len(l) - window + 1):
        yield l[i:i+window]
