"""Independent 80-digit, exact-binomial reference for JS external metrics.

This intentionally uses the original MI/EMI definition, never the JavaScript
conditional-entropy transformation or mode-centered probability recurrence.
Runtime has no dependency on this development-only Python audit.
"""
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
from itertools import permutations
from math import comb
from pathlib import Path
import json
import random
import sys

HERE = Path(__file__).resolve().parent
METHODS = ('arithmetic', 'geometric', 'min', 'max')
D = Decimal


@lru_cache(None)
def ln(n):
    return D(n).ln()


def partitions(n):
    if not n:
        yield []
        return
    def walk(prefix, maximum):
        if len(prefix) == n:
            yield prefix
            return
        for k in range(maximum + 2):
            yield from walk(prefix + [k], max(maximum, k))
    yield from walk([0], 0)


def observed_information(x, y):
    n = len(x)
    a, b = Counter(x), Counter(y)
    cells = Counter(zip(x, y))
    return sum((D(v) / n * (ln(n) + ln(v) - ln(a[i]) - ln(b[j]))
                for (i, j), v in cells.items()), D(0))


@lru_cache(None)
def pair_expectation(n, a, b):
    # Each probability is an exact integer numerator / integer denominator.
    denominator = D(comb(n, b))
    return sum((D(comb(a, k) * comb(n - a, b - k)) / denominator
                * D(k) / n * (ln(n) + ln(k) - ln(a) - ln(b))
                for k in range(max(1, a + b - n), min(a, b) + 1)), D(0))


@lru_cache(None)
def expected_information(n, a, b):
    ac, bc = Counter(a), Counter(b)
    return sum((D(am * bm) * pair_expectation(n, ai, bi)
                for ai, am in ac.items() for bi, bm in bc.items()), D(0))


def exact_ari(x, y):
    n = len(x)
    # Unordered pair counts, rather than the runtime's squared counts.
    a, b, c = Counter(x), Counter(y), Counter(zip(x, y))
    total = comb(n, 2)
    agreed = sum(comb(z, 2) for z in c.values())
    same_a = sum(comb(z, 2) for z in a.values())
    same_b = sum(comb(z, 2) for z in b.values())
    denominator = (same_a + same_b) * total - 2 * same_a * same_b
    if denominator == 0:
        return 1.0
    return float(Fraction(2 * (agreed * total - same_a * same_b), denominator))


def reference(x, y):
    n = len(x)
    a, b, c = Counter(x), Counter(y), Counter(zip(x, y))
    ka, kb = len(a), len(b)
    if not n or (ka == kb == 1) or len(c) == ka == kb:
        return dict.fromkeys(METHODS, 1.0)
    if ka == 1 or kb == 1:
        return dict.fromkeys(METHODS, 0.0)
    if ka == n or kb == n:
        return {m: None if m == 'min' else 0.0 for m in METHODS}
    ha = ln(n) - sum((D(v) / n * ln(v) for v in a.values()), D(0))
    hb = ln(n) - sum((D(v) / n * ln(v) for v in b.values()), D(0))
    mi = observed_information(x, y)
    emi = expected_information(n, tuple(sorted(a.values())), tuple(sorted(b.values())))
    norms = {'arithmetic': (ha + hb) / 2, 'geometric': (ha * hb).sqrt(),
             'min': min(ha, hb), 'max': max(ha, hb)}
    return {m: float((mi - emi) / (value - emi)) for m, value in norms.items()}


def make_fixtures():
    fixtures = []
    def add(name, x, y):
        fixtures.append({'name': name, 'x': list(x), 'y': list(y), 'ari': exact_ari(x, y), 'ami': reference(x, y)})
    for n in range(6):
        parts = list(partitions(n))
        for i, x in enumerate(parts):
            for j, y in enumerate(parts):
                add(f'exhaustive-{n}-{i}-{j}', x, y)
    rng = random.Random(91720261002)
    for n in (7, 17, 31, 64, 127, 257):
        for ka, kb in ((2, 2), (3, 7), (10, 20), (n // 2, n // 3)):
            for shape in ('uniform', 'skewed', 'correlated'):
                if shape == 'skewed':
                    x = [min(ka - 1, int(rng.paretovariate(2)) - 1) for _ in range(n)]
                    y = [min(kb - 1, int(rng.paretovariate(1.4)) - 1) for _ in range(n)]
                else:
                    x = [rng.randrange(ka) for _ in range(n)]
                    y = [rng.randrange(kb) for _ in range(n)]
                    if shape == 'correlated':
                        y = [xx if rng.random() < .85 else yy for xx, yy in zip(x, y)]
                add(f'{shape}-{n}-{ka}-{kb}', x, y)
    for n in (32, 1000, 3000):
        x, y = list(range(n)), list(range(n))
        x[1], y[3] = x[0], y[2]
        add(f'distinct-doublets-{n}', x, y)
        y[4] = y[2]
        add(f'doublet-vs-triple-{n}', x, y)
        x, y = [0] * (n - 1) + [1], [1] + [0] * (n - 1)
        add(f'rare-singletons-{n}', x, y)
    add('strings', ['cat', 'dog', 'cat', 'fox'], ['a', 'a', 'b', 'b'])
    add('safe-integer-extremes', [-(2**53 - 1), 2**53 - 1, -(2**53 - 1), 0], [0, 0, 1, 2])
    return fixtures


def verify_expectation_by_enumeration():
    worst = D(0)
    checked = 0
    # Enumerate complete permutation models independently of the binomial sum.
    for x, y in (([0, 0, 0, 1, 1, 2], [0, 0, 1, 1, 2, 2]),
                 ([0, 0, 1, 1, 2, 2], [0, 0, 0, 0, 1, 2]),
                 ([0, 0, 0, 0, 1, 1], [0, 1, 2, 3, 4, 4])):
        ys = set(permutations(y))
        enumeration = sum((observed_information(x, z) for z in ys), D(0)) / len(ys)
        expectation = expected_information(len(x), tuple(sorted(Counter(x).values())), tuple(sorted(Counter(y).values())))
        error = abs(enumeration - expectation)
        assert error < D('1e-76'), error
        worst = max(worst, error)
        checked += len(ys)
    return {'permutations_checked': checked, 'max_decimal_error': str(worst)}


def sklearn_artifacts():
    import sklearn
    from sklearn.metrics import adjusted_mutual_info_score
    rows = []
    for n in (3, 10, 100, 256, 1000, 3000):
        x, y = list(range(n)), list(range(n))
        y[1] = y[0]
        for m in METHODS:
            rows.append({'case': 'singleton-vs-doublet', 'n': n, 'averageMethod': m,
                         'exact': None if m == 'min' else 0.0,
                         'sklearn': adjusted_mutual_info_score(x, y, average_method=m)})
        if n < 4:
            continue
        x, y = list(range(n)), list(range(n))
        x[1], y[3] = x[0], y[2]
        exact = -2 / (n * (n - 1) - 2)
        for m in METHODS:
            value = adjusted_mutual_info_score(x, y, average_method=m)
            rows.append({'case': 'distinct-doublets', 'n': n, 'averageMethod': m,
                         'exact': exact, 'sklearn': value, 'absolute_error': abs(value - exact)})
    return {'sklearn_version': sklearn.__version__, 'python': sys.version, 'examples': rows}


def synthetic_references():
    ari = []
    for scale in (1, 10_000, 100_000, 13_558_037, 13_558_039, 613_566_756):
        cells = [2 * scale, scale, scale, 3 * scale]
        n = sum(cells)
        a = b = [3 * scale, 4 * scale]
        total = comb(n, 2)
        same_cells = sum(comb(v, 2) for v in cells)
        same_a, same_b = (sum(comb(v, 2) for v in counts) for counts in (a, b))
        value = Fraction(2 * (same_cells * total - same_a * same_b),
                         (same_a + same_b) * total - 2 * same_a * same_b)
        ari.append({'n': n, 'a': a, 'b': b, 'cells': cells, 'ari': float(value),
                    'rational_numerator': str(value.numerator), 'rational_denominator': str(value.denominator)})
    conditional = []
    for n, first, other in ((2**26, 1, 2**26 - 1), (2**26, 2, 2**25),
                            (2**26, 2**26 - 1, 2), (2**26, 2**26 - 1, 2**26 - 2),
                            (2**26, 2**26 - 2, 2**26 - 1), (2**26, 1, 2),
                            (10_000, 11, 4_999), (20_000, 200, 10_001)):
        a, b = sorted((first, other))
        probability_denominator = D(comb(n, a))
        expectation = sum((D(comb(b, k) * comb(n-b, a-k)) / probability_denominator
                           * D(k) / n * (ln(other) - ln(k))
                           for k in range(max(1, a+b-n), a+1)), D(0))
        conditional.append({'n': n, 'first': first, 'other': other, 'expected': float(expectation)})
    ami = []
    for n in (999_999, 1_234_567, 60_000_001, 2**26 - 1, 2**26):
        for cells in ([n-2, 1, 1, 0], [n-2, 1, 0, 1], [n-5, 3, 2, 0], [n-3, 1, 0, 2]):
            a = [sum(cells[:2]), sum(cells[2:])]
            b = [cells[0] + cells[2], cells[1] + cells[3]]
            rows, cols = [0, 0, 1, 1], [0, 1, 0, 1]
            ha = ln(n) - sum((D(v) / n * ln(v) for v in a), D(0))
            hb = ln(n) - sum((D(v) / n * ln(v) for v in b), D(0))
            mi = sum((D(v) / n * (ln(n) + ln(v) - ln(a[r]) - ln(b[c]))
                      for v, r, c in zip(cells, rows, cols) if v), D(0))
            emi = expected_information(n, tuple(sorted(a)), tuple(sorted(b)))
            norms = {'arithmetic': (ha + hb) / 2, 'geometric': (ha * hb).sqrt(),
                     'min': min(ha, hb), 'max': max(ha, hb)}
            values = {m: float((mi - emi) / (value - emi)) for m, value in norms.items()}
            nz = [i for i, v in enumerate(cells) if v]
            ami.append({'n': n, 'a': a, 'b': b, 'cells': [cells[i] for i in nz],
                        'rows': [rows[i] for i in nz], 'cols': [cols[i] for i in nz], 'ami': values})
    return {'ari': ari, 'conditional': conditional, 'ami': ami}


if __name__ == '__main__':
    with localcontext() as ctx:
        ctx.prec = 80
        enumeration = verify_expectation_by_enumeration()
        fixtures = make_fixtures()
        content = {'reference': '80-digit Decimal original MI/EMI formula with exact integer hypergeometric probabilities',
                   'precision_decimal_digits': ctx.prec, 'permutation_validation': enumeration,
                   'fixtures': fixtures}
        (HERE / 'numeric-reference.json').write_text(json.dumps(content, separators=(',', ':')) + '\n')
        artifacts = sklearn_artifacts()
        (HERE / 'sklearn-conditioning.json').write_text(json.dumps(artifacts, indent=2) + '\n')
        (HERE / 'synthetic-reference.json').write_text(json.dumps(synthetic_references(), indent=2) + '\n')
        print(json.dumps({'fixtures': len(fixtures), 'permutation_validation': enumeration,
                          'sklearn_version': artifacts['sklearn_version']}, indent=2))
