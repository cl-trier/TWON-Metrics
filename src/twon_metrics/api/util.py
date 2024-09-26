import typing


def batched(iterable:typing.Iterable, n:int=1) -> typing.Iterator[typing.Any]:
    l: int = len(iterable)

    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]