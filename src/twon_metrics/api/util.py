import typing


def batched(iterable: typing.Iterable, n: int = 1) -> typing.Iterator[typing.Any]:
    for ndx in range(0, len(iterable), n):
        yield iterable[ndx : min(ndx + n, len(iterable))]
