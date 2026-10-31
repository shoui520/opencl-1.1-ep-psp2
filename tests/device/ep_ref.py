"""References for the math tests: what table 10.1 and sections 7.4 and 7.5 allow
a device's single precision result to be.

A reference is the exact result: a Fraction when it is known exactly, an mpf
of a host libm (double) or mpmath result otherwise (its error is far below a
single precision ulp), or a float for the results F.9 of C99 TC2 and section
7.5.1 prescribe (NaN, the infinities, signed zeros, exp(0) = 1 and the
like), which must be produced exactly (7.5.1: "Those values shall produce
exactly the prescribed answers, and no other").  Section 7.5.3's flush to
zero allowances are applied for every function: a sub-normal operand may be
read as either zero and a result sub-normal before rounding may be either
zero."""
import ctypes
import collections
import itertools
import math
from fractions import Fraction

import mpmath as mp
import numpy as np

mp.mp.prec = 160

INF, NAN = math.inf, math.nan
FLT_MAX = float(np.finfo(np.float32).max)
FLT_MIN = 2.0 ** -126
DENORM_MIN = 2.0 ** -149
INT_MAX, INT_MIN = 2 ** 31 - 1, -2 ** 31
TWO128 = Fraction(2) ** 128
FRACT_MAX = float.fromhex('0x1.fffffep-1')
HALF = Fraction(1, 2)

_libm = ctypes.CDLL('libm.so.6')
_T = {'d': ctypes.c_double, 'f': ctypes.c_float, 'p': ctypes.POINTER(ctypes.c_int)}


def _fn(name, args='d', res='d'):
    f = getattr(_libm, name)
    f.argtypes = [_T[a] for a in args]
    f.restype = _T[res]
    return f


LIBM = {n: _fn(n) for n in ('acos acosh asin asinh atan atanh cbrt cos cosh erfc erf exp exp2 '
                            'exp10 expm1 log log2 log10 log1p sin sinh sqrt tan tanh tgamma '
                            'lgamma ceil floor rint round trunc fabs logb').split()}
LIBM.update({n: _fn(n, 'dd') for n in 'atan2 hypot pow copysign fmax fmin fmod remainder'.split()})
_nextafterf = _fn('nextafterf', 'ff', 'f')
_lgamma_r = _fn('lgamma_r', 'dp')


# ---- exact values, rounding and ulps (section 7.4) ----

def is_special(v):
    return isinstance(v, float)


def E(d):
    """The exact value for a host double result: the specials stay floats."""
    if d == 0 or math.isinf(d) or math.isnan(d):
        return float(d)
    return mp.mpf(d)


def frac(v):
    if isinstance(v, Fraction):
        return v
    if isinstance(v, float):
        return Fraction(v)
    sign, man, exp, _ = v._mpf_
    return (-1) ** sign * Fraction(man) * Fraction(2) ** exp


def rnd(v, mode):
    """v (finite, non-zero) correctly rounded to single precision in mode
    rte, rtz, rtp or rtn."""
    f = frac(v)
    s = -1 if f < 0 else 1
    a = abs(f)
    ex = a.numerator.bit_length() - a.denominator.bit_length() + 1
    if a < Fraction(2) ** (ex - 1):
        ex -= 1
    q = Fraction(2) ** max(ex - 24, -149)
    n = a / q
    fl = n.numerator // n.denominator
    rem = n - fl
    up = {'rte': rem > HALF or (rem == HALF and fl % 2 == 1), 'rtz': False,
          'rtp': rem > 0 and s > 0, 'rtn': rem > 0 and s < 0}[mode]
    r = (fl + up) * q
    if r >= TWO128:
        big = {'rte': True, 'rtz': False, 'rtp': s > 0, 'rtn': s < 0}[mode]
        return s * (INF if big else FLT_MAX)
    return math.copysign(float(r), s)


def ulp(v):
    a = abs(mp.mpf(frac(v).numerator) / frac(v).denominator) if isinstance(v, Fraction) else abs(v)
    _, ex = mp.frexp(a)
    return mp.ldexp(1, min(max(ex - 24, -149), 104))


def ulp_err(dev, v):
    e = mp.mpf(v.numerator) / v.denominator if isinstance(v, Fraction) else mp.mpf(v)
    if math.isinf(dev):
        if abs(e) >= mp.ldexp(1, 128) and (dev > 0) == (e > 0):
            return 0.0
        d = mp.ldexp(1, 128) * (1 if dev > 0 else -1)
    else:
        d = mp.mpf(dev)
    return float(abs(d - e) / ulp(e))


def same(a, b):
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b and math.copysign(1, a) == math.copysign(1, b)


# +0 for -0 taken (section 10 item 6), per function, for collect() to report
PLUS_ZERO = collections.Counter()
CURRENT = [None]


def accepts(dev, v, kind, bound=0, modes=('rtz',), zero='plus'):
    """Whether device result dev conforms for the exact result v.
    kind: 'exact' (0 ulp), 'cr' (correctly rounded in one of modes) or 'ulp'
    (within bound).  zero: 'exact' (a zero must have the right sign), 'any'
    (either zero) or 'plus' (+0 is also allowed for -0: section 10 item 6,
    "Zero results may always be positive 0.0"; each such is counted)."""
    if isinstance(v, int):
        return dev == v
    if is_special(v):
        if math.isnan(v):
            return math.isnan(dev)
        if v == 0 and dev == 0:
            if zero == 'any' or math.copysign(1, dev) == math.copysign(1, v):
                return True
            if zero == 'plus' and math.copysign(1, dev) > 0:
                PLUS_ZERO[CURRENT[0]] += 1
                return True
            return False
        if 0 < abs(v) < FLT_MIN and dev == 0:
            return True  # 7.5.3 (2)
        return same(dev, v)
    if math.isnan(dev):
        return False
    if abs(frac(v)) < FLT_MIN and dev == 0:
        return True  # 7.5.3 (2): sub-normal before rounding
    if kind == 'exact':
        return same(dev, rnd(v, 'rte'))
    if kind == 'cr':
        return any(same(dev, rnd(v, m)) for m in modes)
    if 'rtz' in modes and abs(frac(v)) > Fraction(FLT_MAX) and dev == math.copysign(FLT_MAX, frac(v)):
        return True  # an overflow, rounded toward zero
    return ulp_err(dev, v) <= bound


def ftz_variants(args):
    """The operand lists 7.5.3 (3) allows: each sub-normal float operand as is
    or as either zero."""
    opts = []
    for a in args:
        if isinstance(a, float) and 0 < abs(a) < FLT_MIN:
            opts.append((a, 0.0, -0.0))
        else:
            opts.append((a,))
    return list(itertools.product(*opts))


# ---- the functions (section 6.11.2, edge cases of F.9 and 7.5.1) ----

def libm1(name, prescribed=None):
    f = LIBM[name]

    def r(x):
        if prescribed:
            p = prescribed(x)
            if p is not None:
                return p
        return E(f(x))
    return r


def libm2(name):
    f = LIBM[name]
    return lambda x, y: E(f(x, y))


def one_at_zero(x):
    return 1.0 if x == 0 else None


def mp_or_special(v):
    return v if is_special(v) else mp.mpf(v)


def acospi(x):
    if math.isnan(x) or abs(x) > 1:
        return NAN
    if x == 1:
        return 0.0
    return mp.acos(x) / mp.pi


def asinpi(x):
    if math.isnan(x) or abs(x) > 1:
        return NAN
    if x == 0:
        return x
    return mp.asin(x) / mp.pi


def atanpi(x):
    if math.isnan(x):
        return NAN
    if x == 0:
        return x
    if math.isinf(x):
        return math.copysign(0.5, x)
    return mp.atan(x) / mp.pi


def atan2pi(y, x):
    if math.isnan(x) or math.isnan(y):
        return NAN
    neg_x = math.copysign(1, x) < 0
    if y == 0:
        return math.copysign(1.0 if neg_x else 0.0, y)
    if x == 0:
        return math.copysign(0.5, y)
    if math.isinf(y):
        if math.isinf(x):
            return math.copysign(0.75 if neg_x else 0.25, y)
        return math.copysign(0.5, y)
    if math.isinf(x):
        return math.copysign(1.0 if neg_x else 0.0, y)
    return mp.atan2(y, x) / mp.pi


def cospi(x):
    if math.isnan(x) or math.isinf(x):
        return NAN
    if x == 0:
        return 1.0
    r = math.fmod(abs(x), 2.0)
    if r in (0.5, 1.5):
        return 0.0
    return mp.cospi(r)


def sinpi(x):
    if math.isnan(x) or math.isinf(x):
        return NAN
    if x == math.floor(x):
        return math.copysign(0.0, x)
    return mp.sinpi(math.fmod(x, 2.0))


def tanpi(x):
    if math.isnan(x) or math.isinf(x):
        return NAN
    if x == math.floor(x):
        # tanpi(n): copysign(0, n) for even n, copysign(0, -n) for odd n
        odd = math.fmod(x, 2.0) != 0
        return math.copysign(0.0, -x if odd else x)
    n = math.floor(x)
    if x - n == 0.5:
        return -INF if math.fmod(n, 2.0) else INF
    r = math.fmod(x, 2.0)
    return mp.sinpi(r) / mp.cospi(r)


def pow_(x, y):
    if y == 0 or x == 1 or (x == -1 and math.isinf(y)):
        return 1.0
    return E(LIBM['pow'](x, y))


def pown(x, n):
    if n == 0:
        return 1.0
    return E(LIBM['pow'](x, float(n)))


def powr(x, y):
    if math.isnan(x) or math.isnan(y) or x < 0:
        return NAN
    if x == 0:
        if y == 0:
            return NAN
        return INF if y < 0 else 0.0
    if math.isinf(x):
        if y == 0:
            return NAN
        return 0.0 if y < 0 else INF
    if x == 1:
        return NAN if math.isinf(y) else 1.0
    if y == 0:
        return 1.0
    return E(LIBM['pow'](x, y))


def rootn(x, n):
    if n == 0 or math.isnan(x) or (x < 0 and n % 2 == 0):
        return NAN
    if x == 0:
        if n < 0:
            return INF if n % 2 == 0 else math.copysign(INF, x)
        return 0.0 if n % 2 == 0 else x
    if math.isinf(x):
        return math.copysign(INF if n > 0 else 0.0, x)
    v = mp.power(abs(x), mp.mpf(1) / n)
    return -v if x < 0 else v


def rsqrt(x):
    if math.isnan(x) or x < 0:
        return NAN
    if x == 0:
        return math.copysign(INF, x)
    if math.isinf(x):
        return 0.0
    return 1 / mp.sqrt(x)


def arith(op):
    """x + y, x - y, x * y and x / y: exact, IEEE 754 specials."""
    def r(x, y):
        with np.errstate(all='ignore'):
            d = float({'add': np.add, 'sub': np.subtract, 'mul': np.multiply,
                       'div': np.divide}[op](np.float64(x), np.float64(y)))
        if not (math.isfinite(x) and math.isfinite(y)) or math.isnan(d) or math.isinf(d):
            return d
        fx, fy = Fraction(x), Fraction(y)
        v = {'add': lambda: fx + fy, 'sub': lambda: fx - fy, 'mul': lambda: fx * fy,
             'div': lambda: fx / fy}[op]()
        return d if v == 0 else v  # a zero's sign is IEEE's (the double's)
    return r


def recip(x):
    return arith('div')(1.0, x)


def fma_(x, y, z):
    if math.isnan(x) or math.isnan(y) or math.isnan(z):
        return NAN
    if math.isinf(x) or math.isinf(y):
        if x == 0 or y == 0:
            return NAN
        p = math.copysign(INF, x) * math.copysign(1, y)
        if math.isinf(z) and z != p:
            return NAN
        return p
    if math.isinf(z):
        return z
    v = Fraction(x) * Fraction(y) + Fraction(z)
    if v == 0:
        prod_neg = (math.copysign(1, x) * math.copysign(1, y)) < 0
        both_neg = prod_neg and math.copysign(1, z) < 0 and x * y == 0 and z == 0
        return -0.0 if both_neg else 0.0
    return v


def mad(x, y, z):
    return None  # table 10.1: any value


def fdim(x, y):
    if math.isnan(x) or math.isnan(y):
        return NAN
    if not x > y:
        return 0.0
    if math.isinf(x) or math.isinf(y):
        return INF
    return Fraction(x) - Fraction(y)


def fract(x):
    """(fract, *iptr): fmin(x - floor(x), 0x1.fffffep-1f), floor(x)."""
    if math.isnan(x):
        return NAN, NAN
    if math.isinf(x):
        return math.copysign(0.0, x), x
    if x == 0:
        return x, x
    fl = math.floor(x)
    v = Fraction(x) - fl
    return (FractClamp(v) if v else 0.0), float(fl)


class FractClamp(Fraction):
    """x - floor(x), rounded, then fmin with 0x1.fffffep-1f."""


def fmax_(x, y):
    return E(LIBM['fmax'](x, y))


def fmin_(x, y):
    return E(LIBM['fmin'](x, y))


def maxmag(x, y):
    if abs(x) > abs(y):
        return E(x)
    if abs(y) > abs(x):
        return E(y)
    return fmax_(x, y)


def minmag(x, y):
    if abs(x) < abs(y):
        return E(x)
    if abs(y) < abs(x):
        return E(y)
    return fmin_(x, y)


def ldexp_(x, n):
    if x == 0 or math.isnan(x) or math.isinf(x):
        return x
    return Fraction(x) * Fraction(2) ** n


def nextafter(x, y):
    return E(_nextafterf(x, y))


def nextafter_ftz(x, y):
    """The results 7.5.3 allows instead of 7.5.1's when flushing."""
    if x == FLT_MIN and y < FLT_MIN:
        return 0.0
    if x == -FLT_MIN and y > -FLT_MIN:
        return -0.0
    if x == 0 and y > 0:
        return FLT_MIN
    if x == 0 and y < 0:
        return -FLT_MIN
    return None


def frexp_(x):
    if math.isnan(x) or math.isinf(x) or x == 0:
        return x, 0
    m, e = math.frexp(x)
    return E(m), e


def ilogb(x):
    """The allowed results (a set): FP_ILOGB0 is INT_MIN or -INT_MAX, FP_ILOGBNAN
    INT_MAX or INT_MIN (6.11.2.1)."""
    if math.isnan(x):
        return {INT_MAX, INT_MIN}
    if x == 0:
        return {INT_MIN, -INT_MAX}
    if math.isinf(x):
        return {INT_MAX}
    return {math.frexp(x)[1] - 1}


def modf_(x):
    """7.5.2: *iptr = trunc(x); copysign(isinf(x) ? 0.0 : x - *iptr, x)."""
    if math.isnan(x):
        return NAN, NAN
    ip = math.copysign(float(math.trunc(x)), x) if math.isfinite(x) else x
    f = 0.0 if math.isinf(x) else x - ip
    f = math.copysign(f, x) if f == 0 else f
    return E(f), E(ip)


def remquo_(x, y):
    """(r, quo): 7.5.1 and 6.11.2, quo the low seven bits of the quotient with
    the sign of x/y."""
    if math.isnan(x) or math.isnan(y) or math.isinf(x) or y == 0:
        return NAN, 0
    r = E(LIBM['remainder'](x, y))
    if math.isinf(y):
        return r, 0
    q = Fraction(x) / Fraction(y)
    k = math.floor(q)
    if q - k > HALF or (q - k == HALF and k % 2):
        k += 1
    m = abs(k) & 127
    return r, (-m if q < 0 else m)


def sincos(x):
    return libm1('sin')(x), libm1('cos')(x)


def lgamma_r(x):
    s = ctypes.c_int(0)
    v = _lgamma_r(x, ctypes.byref(s))
    return E(v), s.value


# name: (reference, kind, bound); kind 'exact' (0 ulp), 'cr', 'ulp', 'report'
# (lgamma: table 10.1 has no bound, 7.4's footnote 59: undefined) or 'none'
# (mad, native_*).  Table 10.1's values.
def U(n):
    return ('ulp', n)


FUNCS = {
    'acos': (libm1('acos'), U(4)), 'acospi': (acospi, U(5)), 'asin': (libm1('asin'), U(4)),
    'asinpi': (asinpi, U(5)), 'atan': (libm1('atan'), U(5)), 'atan2': (libm2('atan2'), U(6)),
    'atanpi': (atanpi, U(5)), 'atan2pi': (atan2pi, U(6)), 'acosh': (libm1('acosh'), U(4)),
    'asinh': (libm1('asinh'), U(4)), 'atanh': (libm1('atanh'), U(5)),
    'cbrt': (libm1('cbrt'), U(4)), 'ceil': (libm1('ceil'), ('cr', 0)),
    'copysign': (libm2('copysign'), ('exact', 0)), 'cos': (libm1('cos', one_at_zero), U(4)),
    'cosh': (libm1('cosh', one_at_zero), U(4)), 'cospi': (cospi, U(4)),
    'erfc': (libm1('erfc', lambda x: {-INF: 2.0, INF: 0.0}.get(x)), U(16)),
    'erf': (libm1('erf', lambda x: math.copysign(1.0, x) if math.isinf(x) else None), U(16)),
    'exp': (libm1('exp', one_at_zero), U(4)), 'exp2': (libm1('exp2', one_at_zero), U(4)),
    'exp10': (libm1('exp10', one_at_zero), U(4)),
    'expm1': (libm1('expm1', lambda x: -1.0 if x == -INF else None), U(4)),
    'fabs': (libm1('fabs'), ('exact', 0)), 'fdim': (fdim, ('cr', 0)),
    'floor': (libm1('floor'), ('cr', 0)), 'fma': (fma_, ('cr', 0)),
    'fmax': (fmax_, ('exact', 0)), 'fmin': (fmin_, ('exact', 0)),
    'fmod': (libm2('fmod'), ('exact', 0)), 'hypot': (libm2('hypot'), U(4)),
    'ldexp': (ldexp_, ('cr', 0)), 'lgamma': (libm1('lgamma'), ('report', 0)),
    'log': (libm1('log'), U(4)), 'log2': (libm1('log2'), U(4)), 'log10': (libm1('log10'), U(4)),
    'log1p': (libm1('log1p'), U(4)), 'logb': (libm1('logb'), ('exact', 0)),
    'mad': (mad, ('none', 0)), 'maxmag': (maxmag, ('exact', 0)),
    'minmag': (minmag, ('exact', 0)), 'nextafter': (nextafter, ('exact', 0)),
    'pow': (pow_, U(16)), 'pown': (pown, U(16)), 'powr': (powr, U(16)),
    'remainder': (libm2('remainder'), ('exact', 0)), 'rint': (libm1('rint'), ('cr', 0)),
    'rootn': (rootn, U(16)), 'round': (libm1('round'), ('cr', 0)), 'rsqrt': (rsqrt, U(4)),
    'sin': (libm1('sin'), U(4)), 'sinh': (libm1('sinh'), U(4)), 'sinpi': (sinpi, U(4)),
    'sqrt': (libm1('sqrt'), U(4)), 'tan': (libm1('tan'), U(5)), 'tanh': (libm1(
        'tanh', lambda x: math.copysign(1.0, x) if math.isinf(x) else None), U(5)),
    'tanpi': (tanpi, U(6)), 'tgamma': (libm1('tgamma'), U(16)),
    'trunc': (libm1('trunc'), ('cr', 0)),
    'add': (arith('add'), ('cr', 0)), 'sub': (arith('sub'), ('cr', 0)),
    'mul': (arith('mul'), ('cr', 0)), 'div': (arith('div'), U(3)), 'recip': (recip, U(3)),
}
# 7.5.1: half_ functions have the edge cases of the function without the
# prefix, and up to 8192 ulp otherwise
HALF_OF = {'half_cos': 'cos', 'half_divide': 'div', 'half_exp': 'exp', 'half_exp2': 'exp2',
           'half_exp10': 'exp10', 'half_log': 'log', 'half_log2': 'log2',
           'half_log10': 'log10', 'half_powr': 'powr', 'half_recip': 'recip',
           'half_rsqrt': 'rsqrt', 'half_sin': 'sin', 'half_sqrt': 'sqrt', 'half_tan': 'tan'}
for _h, _f in HALF_OF.items():
    FUNCS[_h] = (FUNCS[_f][0], U(8192))
for _h in HALF_OF:
    FUNCS['native_' + _h[5:]] = (None, ('none', 0))
# both zeros are fine where the result is either of two zero operands, and
# section 10 item 6: "Zero results may always be positive 0.0"
ZERO = {'fmax': 'any', 'fmin': 'any', 'maxmag': 'any', 'minmag': 'any',
        'add': 'plus', 'sub': 'plus', 'mul': 'plus',
        # no prescribed zeros: they are underflows, either sign within bounds
        'tgamma': 'any'}


def check_value(name, args, dev, modes):
    """None if dev conforms for name(args), else the reason."""
    ref, (kind, bound) = FUNCS[name]
    zero = ZERO.get(name, 'plus')
    CURRENT[0] = name
    want = []
    for a in ftz_variants(args):
        v = ref(*a)
        want.append(v)
        if isinstance(v, FractClamp):
            ok = any(same(dev, min(rnd(v, m), FRACT_MAX)) for m in modes)
        else:
            ok = accepts(dev, v, kind, bound, modes, zero)
        if ok:
            return None
    if name == 'nextafter':
        alt = nextafter_ftz(*args)
        if alt is not None and same(dev, alt):
            return None
    v = want[0]
    if kind == 'ulp' and not is_special(v) and math.isfinite(dev):
        return 'want %s, %.1f ulp' % (mp.nstr(mp.mpf(frac(v).numerator) / frac(v).denominator, 9)
                                       if isinstance(v, Fraction) else mp.nstr(v, 9),
                                       ulp_err(dev, v))
    if isinstance(v, FractClamp):
        return 'want %r' % min(rnd(v, modes[0]), FRACT_MAX)
    if not is_special(v) and kind in ('cr', 'exact'):
        return 'want %r' % rnd(v, modes[0] if kind == 'cr' else 'rte')
    return 'want %r' % (v,)


def max_ulp(name, args_list, devs):
    """The largest error over finite references, for 'report' functions."""
    ref = FUNCS[name][0]
    worst = 0.0
    for args, dev in zip(args_list, devs):
        v = ref(*args)
        if not is_special(v) and not math.isnan(dev):
            worst = max(worst, ulp_err(dev, v))
    return worst
