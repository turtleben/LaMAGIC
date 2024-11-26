from itertools import combinations


def powerset(iterable):
    """
    Return powerset except empty set
    """
    s = list(iterable)
    for r in range(1, len(s) + 1):
        for _ in combinations(s, r):
            yield _
