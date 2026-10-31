"""Tests of what the OpenCL 1.1 specification (revision 44) requires of an
embedded profile device, section by section: section 10 lists how the
embedded profile differs from the full profile; everything else is as the
earlier sections describe.  Kernels are tests/kernels/ep_*.cl."""
import math
import os
import zlib

import numpy as np

import ep_ref
from devtest import EP_API_REPORT, ApiCase, Buf, Case, Local

K = 'tests/kernels/'
F32 = np.float32
MODES = ('rtz',)  # section 10 item 5: the device has CL_FP_ROUND_TO_ZERO only


def ep_api_script(s, rng):
    s.add('ep_api', compare=False)


def ep_api_check(outs):
    out = outs[0].tobytes()
    failures = int(np.frombuffer(out[:4], np.int32)[0])
    text = out[4:4 + EP_API_REPORT].split(b'\0')[0].decode(errors='replace')
    if not failures:
        return []
    return ['%d API checks failed:' % failures] + ['  ' + line for line in text.splitlines()]


# ---- section 6.11.2: the math functions, against table 10.1 and section 7.5 ----

N = 1024
SPECIALS = [0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 2.0,
            -2.0, 3.0, -3.0, 0.25, -0.25, 0.75, 10.0, -10.0, 100.0, -100.0, ep_ref.FLT_MAX,
            -ep_ref.FLT_MAX, ep_ref.FLT_MIN, -ep_ref.FLT_MIN, ep_ref.DENORM_MIN,
            -ep_ref.DENORM_MIN, float.fromhex('0x0.fffffep-126'), ep_ref.FRACT_MAX,
            float.fromhex('0x1.000002p0'), 2.0 ** 24, -2.0 ** 24, 2.0 ** 24 + 2, 8388609.0,
            8388608.5, 1e10, -1e10, 1e-10, -1e-10, 1e30, -1e30, math.pi, -math.pi, math.pi / 2,
            2.0 ** 16, -2.0 ** 16, 65535.5, 88.72, -87.33, 89.0, -104.0, 126.0, 128.0, -126.0,
            -150.0, 0.1, -0.1, 7.5, -7.5, 1e-38, -1e-38]
# two operands: these crossed
PAIRS = [0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.0, -3.0, 2.5,
         ep_ref.FLT_MAX, ep_ref.FLT_MIN, ep_ref.DENORM_MIN, -ep_ref.DENORM_MIN, 1e-10, 100.0]
# three operands: these crossed
TRIPLES = [0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, -1.5, ep_ref.DENORM_MIN]
NS = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 7, -7, 10, -10, 24, -24, 127, -127, 1000]

# where the uniform half of the random operands is, (lo, hi), by function or
# by the function a half_ or native_ one is a version of
DOMAIN = {
    'acos': (-1, 1), 'asin': (-1, 1), 'acospi': (-1, 1), 'asinpi': (-1, 1), 'atanh': (-1, 1),
    'acosh': (1, 1e4), 'erf': (-5, 5), 'erfc': (-5, 12), 'tanh': (-10, 10),
    'cospi': (-10, 10), 'sinpi': (-10, 10), 'tanpi': (-10, 10), 'cosh': (-90, 90),
    'sinh': (-90, 90), 'exp': (-104, 89), 'exp2': (-150, 128), 'exp10': (-45, 38.6),
    'expm1': (-20, 89), 'log': (0, 1e5), 'log2': (0, 1e5), 'log10': (0, 1e5),
    'log1p': (-1, 100), 'sqrt': (0, 1e6), 'rsqrt': (0, 1e6), 'recip': (-1e6, 1e6),
    'tgamma': (-10, 35), 'lgamma': (-10, 100), 'hypot': (-1e20, 1e20), 'pow': (0, 20),
    'powr': (0, 20), 'atan2': (-10, 10), 'atan2pi': (-10, 10), 'nextafter': (-10, 10),
    'ldexp': (-1e3, 1e3), 'pown': (-10, 10), 'rootn': (-1e4, 1e4), 'fmod': (-1e4, 1e4),
    'remainder': (-1e4, 1e4), 'fabs': (-1e6, 1e6), 'copysign': (-1e6, 1e6),
    'logb': (-1e6, 1e6), 'frexp': (-1e6, 1e6), 'ilogb': (-1e6, 1e6),
    'cos': (-100, 100), 'sin': (-100, 100), 'tan': (-100, 100), 'sincos': (-100, 100),
}
DOMAIN_HALF = {'cos': (-2 ** 16, 2 ** 16), 'sin': (-2 ** 16, 2 ** 16), 'tan': (-2 ** 16, 2 ** 16)}
SECOND = {'pow': (-30, 30), 'powr': (-30, 30), 'fmod': (-100, 100), 'remainder': (-100, 100),
          'remquo': (-100, 100)}
INT_SECOND = {'ldexp': (-300, 300), 'pown': (-20, 20), 'rootn': (-10, 10)}
TWO = set('atan2 atan2pi copysign fdim fmax fmin fmod hypot maxmag minmag nextafter pow powr '
          'remainder remquo add sub mul div'.split()) | set(INT_SECOND)
THREE = {'fma', 'mad'}


def plain(name):
    """The function a half_ or native_ one is a version of."""
    for p in ('half_', 'native_'):
        if name.startswith(p):
            return {'divide': 'div'}.get(name[len(p):], name[len(p):])
    return name


def arity(name):
    f = plain(name)
    return 3 if f in THREE else 2 if f in TWO else 1


def operand(rng, n, lo, hi):
    """Half uniform in [lo, hi], half any bit pattern."""
    u = rng.uniform(lo, hi, n - n // 2).astype(F32)
    bits = rng.integers(0, 2 ** 32, n // 2, dtype=np.uint64).astype(np.uint32).view(F32)
    v = np.concatenate([u, bits])
    rng.shuffle(v)
    return v


def inputs(name):
    """The operands, N of each: the specials, crossed when there are two or
    three operands, then random ones."""
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    f = plain(name)
    lo_hi = (DOMAIN_HALF if name.startswith('half_') else {}).get(f) or DOMAIN.get(f, (-100, 100))
    n_ops = arity(name)
    if n_ops == 1:
        s = np.array(SPECIALS, F32)
        return [np.concatenate([s, operand(rng, N - len(s), *lo_hi)])]
    ops = []
    for k in range(n_ops):
        if n_ops == 2:
            n = len(PAIRS) ** 2
            s = [PAIRS[j // len(PAIRS) if k == 0 else j % len(PAIRS)] for j in range(n)]
        else:
            n = len(TRIPLES) ** 3
            s = [TRIPLES[(j >> (3 * (2 - k))) & 7] for j in range(n)]
        if k == 1 and f in INT_SECOND:
            s = np.array([NS[j % len(NS)] for j in range(n)], np.int32)
            r = rng.integers(*INT_SECOND[f], N - n, endpoint=True).astype(np.int32)
        else:
            s = np.array(s, F32)
            r = operand(rng, N - n, *(SECOND.get(f) if k and f in SECOND else lo_hi))
        ops.append(np.concatenate([s, r]))
    if f in THREE:
        # cancellation: z = -(x * y) rounded, and its neighbours towards zero
        x, y, z = ops
        n = len(TRIPLES) ** 3
        with np.errstate(all='ignore'):
            z[n:n + 192] = -(x[n:n + 192] * y[n:n + 192])
            z[n + 128:n + 192] = np.nextafter(z[n + 128:n + 192], F32(0))
    return ops


def py(v):
    return v.item() if isinstance(v, np.generic) else v


def columns(ins, n):
    cols = [ins[k].reshape(-1) for k in range(n)]
    return [tuple(py(c[i]) for c in cols) for i in range(len(cols[0]))]


def collect(name, rows):
    """errors from (args, device result, reason or None) rows"""
    errs, bad = [], 0
    for args, got, why in rows:
        if why:
            bad += 1
            if bad <= 6:
                errs.append('%s(%s) = %r: %s' % (name, ', '.join(map(repr, args)), got, why))
    if bad:
        errs.append('%s: %d of %d wrong' % (name, bad, len(rows)))
    plus = ep_ref.PLUS_ZERO.pop(name, 0)
    if plus:
        print('    %s: +0 for -0 in %d results (section 10 item 6)' % (name, plus))
    return errs


def valid(name, args):
    """Table 6.9's input ranges of half_cos, half_sin, half_tan and half_powr."""
    if name in ('half_cos', 'half_sin', 'half_tan'):
        return math.isfinite(args[0]) and abs(args[0]) <= 2 ** 16
    if name == 'half_powr':
        return args[0] >= 0
    return True


def math_check(name, n_ops):
    def check(ins, ref, dev):
        out = [py(v) for v in dev[n_ops].reshape(-1)]
        rows = [(a, d, ep_ref.check_value(name, a, d, MODES))
                for a, d in zip(columns(ins, n_ops), out) if valid(name, a)]
        return collect(name, rows)
    return check


def report_check(name, n_ops):
    """lgamma: table 10.1 gives no bound (7.4, footnote 59), so the largest
    error is only reported."""
    def check(ins, ref, dev):
        worst = ep_ref.max_ulp(name, columns(ins, n_ops), [py(v) for v in dev[n_ops].reshape(-1)])
        print('    %s: at most %.1f ulp (no bound: 7.4, footnote 59)' % (name, worst))
        return []
    return check


def math_case(name, vec=False):
    kname = ('v_' if vec else 'm_') + name
    ops = inputs(name)
    kind = ep_ref.FUNCS[name][1][0]
    check = {'none': None, 'report': report_check}.get(kind, math_check)
    if check:
        check = check(name, len(ops))
    args = [Buf(o.dtype, N, values=o) for o in ops] + [Buf('f4', N, 'zeros')]
    # mad and native_ have no accuracy to check (table 10.1): that they run
    return Case('ep_' + kname, K + 'ep_math.cl', kname, (N // 4 if vec else N,), args,
                check=check, skip_compare=(len(ops),))


def multi_case(name, refname, outs, n_ops=1):
    """A function with a second result through a pointer: ref(args) gives both;
    outs, the kinds of the results ('f' with the function's bound, or 'i')."""
    ops = inputs(name)[:n_ops]

    def check(ins, ref, dev):
        rows = []
        ep_ref.CURRENT[0] = name
        kind, bound = ep_ref.FUNCS.get(name, (None, ('exact', 0)))[1]
        for i, a in enumerate(columns(ins, n_ops)):
            got = tuple(py(dev[n_ops + k].reshape(-1)[i]) for k in range(len(outs)))
            why = None
            for v in ep_ref.ftz_variants(a):
                want = refname(*v)
                if all(ep_ref.accepts(g, w, kind, bound, MODES)
                       if not isinstance(w, ep_ref.FractClamp) else
                       any(ep_ref.same(g, min(ep_ref.rnd(w, m), ep_ref.FRACT_MAX)) for m in MODES)
                       for g, w in zip(got, want)):
                    break
            else:
                want = refname(*a)
                why = 'want %s' % ', '.join(
                    repr(w) if ep_ref.is_special(w) or isinstance(w, int)
                    else repr(min(ep_ref.rnd(w, MODES[0]), ep_ref.FRACT_MAX))
                    if isinstance(w, ep_ref.FractClamp) else repr(ep_ref.rnd(w, 'rte'))
                    for w in want)
            rows.append((a, got, why))
        return collect(name, rows)
    args = [Buf('f4', N, values=o) for o in ops] + [Buf(t, N, 'zeros') for t in outs]
    return Case('ep_m_' + name, K + 'ep_math.cl', 'm_' + name, (N,), args, check=check,
                skip_compare=tuple(range(n_ops, n_ops + len(outs))))


def sincos_ref(x):
    return ep_ref.libm1('sin')(x), ep_ref.libm1('cos')(x)


ep_ref.FUNCS['sincos'] = (sincos_ref, ('ulp', 4))


def ilogb_case():
    ops = inputs('ilogb')

    def check(ins, ref, dev):
        rows = []
        for a, d in zip(columns(ins, 1), dev[1].reshape(-1)):
            ok = any(int(d) in ep_ref.ilogb(v[0]) for v in ep_ref.ftz_variants(a))
            want = ' or '.join(map(str, sorted(ep_ref.ilogb(a[0]))))
            rows.append((a, int(d), None if ok else 'want ' + want))
        return collect('ilogb', rows)
    return Case('ep_m_ilogb', K + 'ep_math.cl', 'm_ilogb', (N,),
                [Buf('f4', N, values=ops[0]), Buf('i4', N, 'zeros')], check=check,
                skip_compare=(1,))


def nan_case():
    def check(ins, ref, dev):
        d = dev[1].reshape(-1)
        rows = [((int(u),), py(v), None if math.isnan(v) else 'want a NaN')
                for u, v in zip(ins[0].reshape(-1), d)]
        return collect('nan', rows)
    codes = np.random.default_rng(1).integers(0, 2 ** 32, N, dtype=np.uint64).astype(np.uint32)
    codes[:4] = [0, 1, 0x3fffff, 0xffffffff]
    return Case('ep_m_nan', K + 'ep_math.cl', 'm_nan', (N,),
                [Buf('u4', N, values=codes), Buf('f4', N, 'zeros')], check=check,
                skip_compare=(1,))


def fract_spaces_case():
    """fract's pointer to __global and to __local memory (6.11.2): y = fract(x)
    twice, and both pointers' floor(x)."""
    ops = inputs('fract')

    def check(ins, ref, dev):
        rows = []
        for a, y, ip, lp in zip(columns(ins, 1), dev[1].reshape(-1), dev[2].reshape(-1),
                                dev[3].reshape(-1)):
            x = a[0]
            got = py(y), py(ip), py(lp)
            for v in ep_ref.ftz_variants(a):
                f, fl = ep_ref.fract(v[0])
                f = (min(ep_ref.rnd(f, MODES[0]), ep_ref.FRACT_MAX)
                     if isinstance(f, ep_ref.FractClamp) else f)
                # the sum of two zeros is either zero (section 10 item 6)
                if (ep_ref.same(got[0], 2 * f) or (f == 0 and got[0] == 0)) and \
                        ep_ref.same(got[1], fl) and ep_ref.same(got[2], fl):
                    break
            else:
                rows.append((a, got, 'want %r, %r, %r' % (2 * f, fl, fl)))
                continue
            rows.append((a, got, None))
        return collect('fract_spaces', rows)
    return Case('ep_m_fract_spaces', K + 'ep_math.cl', 'm_fract_spaces', (N,),
                [Buf('f4', N, values=ops[0]), Buf('f4', N, 'zeros'), Buf('f4', N, 'zeros'),
                 Buf('f4', N, 'zeros'), Local(4)], check=check, skip_compare=(1, 2, 3))


def scalar_forms_case():
    """fmax, fmin and ldexp of a float4 with a scalar second operand (table 6.8)."""
    rng = np.random.default_rng(7)
    n = N // 4
    x = inputs('fmax')[0]
    s = rng.uniform(-140, 140, n).astype(F32)
    s[:6] = [0.0, -0.0, 1.5, -1.5, 127.0, -149.0]

    def check(ins, ref, dev):
        xs, ss = ins[0].reshape(-1, 4), ins[1].reshape(-1)
        y = dev[2].reshape(-1, 3, 4)
        rows = []
        for i in range(n):
            for c in range(4):
                a = (py(xs[i, c]), py(ss[i]))
                for k, f in enumerate(('fmax', 'fmin')):
                    why = ep_ref.check_value(f, a, py(y[i, k, c]), MODES)
                    rows.append(((f,) + a, py(y[i, k, c]), why))
                a = (py(xs[i, c]), int(ss[i]))
                why = ep_ref.check_value('ldexp', a, py(y[i, 2, c]), MODES)
                rows.append((('ldexp',) + a, py(y[i, 2, c]), why))
        return collect('scalar_forms', rows)
    return Case('ep_v_scalar_forms', K + 'ep_math.cl', 'v_scalar_forms', (n,),
                [Buf('f4', N, values=x), Buf('f4', n, values=s), Buf('f4', 3 * N, 'zeros')],
                check=check, skip_compare=(2,))


def lgamma_r_case():
    """lgamma_r: the value has no bound (7.4, footnote 59) and is reported; the
    sign of gamma(x) it stores must be right where gamma(x) is finite."""
    ops = inputs('lgamma')

    def check(ins, ref, dev):
        args = columns(ins, 1)
        worst = ep_ref.max_ulp('lgamma', args, [py(v) for v in dev[1].reshape(-1)])
        print('    lgamma_r: at most %.1f ulp (no bound: 7.4, footnote 59)' % worst)
        rows = []
        for a, s in zip(args, dev[2].reshape(-1)):
            x = a[0]
            if not math.isfinite(x) or (x <= 0 and x == math.floor(x)) or abs(x) < ep_ref.FLT_MIN:
                continue
            want = ep_ref.lgamma_r(x)[1]
            rows.append((a, int(s), None if int(s) == want else 'want sign %d' % want))
        return collect('lgamma_r', rows)
    return Case('ep_m_lgamma_r', K + 'ep_math.cl', 'm_lgamma_r', (N,),
                [Buf('f4', N, values=ops[0]), Buf('f4', N, 'zeros'), Buf('i4', N, 'zeros')],
                check=check, skip_compare=(1, 2))


# 6.11.2.1: the macros' values, which "must use the values specified"; the
# constants "accurate within the precision of the float type"
CONSTANT_VALUES = [
    ep_ref.FLT_MAX, math.inf, math.inf, math.nan, math.e, math.log2(math.e), math.log10(math.e),
    math.log(2), math.log(10), math.pi, math.pi / 2, math.pi / 4, 1 / math.pi, 2 / math.pi,
    2 / math.sqrt(math.pi), math.sqrt(2), math.sqrt(0.5), ep_ref.FLT_MAX, ep_ref.FLT_MIN,
    2.0 ** -23]
CONSTANT_NAMES = ('MAXFLOAT HUGE_VALF INFINITY NAN M_E_F M_LOG2E_F M_LOG10E_F M_LN2_F M_LN10_F '
                  'M_PI_F M_PI_2_F M_PI_4_F M_1_PI_F M_2_PI_F M_2_SQRTPI_F M_SQRT2_F '
                  'M_SQRT1_2_F FLT_MAX FLT_MIN FLT_EPSILON').split()
INT_CONSTANTS = [('FLT_DIG', {6}), ('FLT_MANT_DIG', {24}), ('FLT_MAX_10_EXP', {38}),
                 ('FLT_MAX_EXP', {128}), ('FLT_MIN_10_EXP', {-37}), ('FLT_MIN_EXP', {-125}),
                 ('FLT_RADIX', {2}), ('FP_ILOGB0', {ep_ref.INT_MIN, -ep_ref.INT_MAX}),
                 ('FP_ILOGBNAN', {ep_ref.INT_MAX, ep_ref.INT_MIN}), ('FP_FAST_FMAF', {0, 1})]


def constants_check(ins, ref, dev):
    errs = []
    f, n = dev[0].reshape(-1), dev[1].reshape(-1)
    for name, want, got in zip(CONSTANT_NAMES, CONSTANT_VALUES, map(py, f)):
        if math.isnan(want) or math.isinf(want) or want in (ep_ref.FLT_MAX, ep_ref.FLT_MIN):
            ok = ep_ref.same(got, want)
        else:
            ok = ep_ref.ulp_err(got, ep_ref.mp.mpf(want)) <= 1
        if not ok:
            errs.append('%s = %r, want %r' % (name, got, F32(want)))
    for (name, want), got in zip(INT_CONSTANTS, n):
        if int(got) not in want:
            errs.append('%s = %d, want %s' % (name, got, ' or '.join(map(str, sorted(want)))))
    return errs


MULTI = [('fract', ep_ref.fract, ('f4', 'f4')), ('frexp', ep_ref.frexp_, ('f4', 'i4')),
         ('modf', ep_ref.modf_, ('f4', 'f4')), ('sincos', sincos_ref, ('f4', 'f4'))]
ep_ref.FUNCS.setdefault('fract', (ep_ref.fract, ('cr', 0)))
ep_ref.FUNCS.setdefault('frexp', (ep_ref.frexp_, ('exact', 0)))
ep_ref.FUNCS.setdefault('modf', (ep_ref.modf_, ('exact', 0)))
ep_ref.FUNCS.setdefault('remquo', (ep_ref.remquo_, ('exact', 0)))
ep_ref.FUNCS.setdefault('lgamma_r', (ep_ref.lgamma_r, ('report', 0)))
SINGLE = [n for n, (_, (kind, _)) in ep_ref.FUNCS.items()
          if n not in ('fract', 'frexp', 'modf', 'remquo', 'sincos', 'lgamma_r')]

EP_CASES = [
    # chapters 4 and 5 and section 10's table 4.3, checked by the harness itself
    ApiCase('ep_api', K + 'ep_api.cl', ep_api_script, check=ep_api_check),
]
EP_CASES += [math_case(n) for n in SINGLE]
EP_CASES += [math_case(n, vec=True)
             for n in 'sin exp log sqrt floor atan pow fmod atan2 fmax'.split()]
EP_CASES += [multi_case(n, r, o) for n, r, o in MULTI]
EP_CASES += [multi_case('remquo', ep_ref.remquo_, ('f4', 'i4'), n_ops=2), ilogb_case(),
             nan_case(), fract_spaces_case(), scalar_forms_case(), lgamma_r_case(),
             Case('ep_m_constants', K + 'ep_math.cl', 'm_constants', (1,),
                  [Buf('f4', 20, 'zeros'), Buf('i4', 10, 'zeros')], check=constants_check,
                  skip_compare=(0, 1))]


# ---- section 6.2.3: explicit conversions; 6.2.4: as_type ----

from fractions import Fraction  # noqa: E402

# name: (dtype, least, greatest); section 10 item 1: no long without cles_khr_int64
CV_TYPES = {'char': ('i1', -2 ** 7, 2 ** 7 - 1), 'uchar': ('u1', 0, 2 ** 8 - 1),
            'short': ('i2', -2 ** 15, 2 ** 15 - 1), 'ushort': ('u2', 0, 2 ** 16 - 1),
            'int': ('i4', -2 ** 31, 2 ** 31 - 1), 'uint': ('u4', 0, 2 ** 32 - 1),
            'float': ('f4', None, None)}
INT_VARIANTS = [(False, 'rtz'), (True, 'rtz'), (False, 'rte'), (False, 'rtz'), (False, 'rtp'),
                (False, 'rtn'), (True, 'rte'), (True, 'rtz'), (True, 'rtp'), (True, 'rtn')]
FLOAT_VARIANTS = [None, 'rte', 'rtz', 'rtp', 'rtn']  # None: the default (6.2.3.2)
INT_EDGES = [0, 1, -1, 2, 127, 128, -128, -129, 255, 256, 32767, 32768, -32768, -32769, 65535,
             65536, 2 ** 24 - 1, 2 ** 24, 2 ** 24 + 1, 2 ** 24 + 3, -2 ** 24 - 1, 2 ** 25 + 3,
             2 ** 31 - 1, -2 ** 31, 2 ** 31 - 64, 2 ** 31 - 65, 2 ** 31, 2 ** 32 - 1, 2 ** 32 - 129,
             2 ** 32 - 128, 2 ** 32 - 127, 123456789, -123456789]
FLOAT_EDGES = [np.nan, np.inf, -np.inf, 0.0, -0.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 0.49999997,
               -0.49999997, 0.50000006, 1.0, -1.0, 127.4, 127.5, 127.6, 128.0, 128.5, -128.5,
               -128.6, -129.0, 255.49998, 255.5, 256.0, -1.5, 32767.5, 32768.0, -32768.5,
               -32769.0, 65535.5, 65536.0, 16777216.0, 2147483520.0, 2147483648.0, -2147483648.0,
               -2147483904.0, 4294967040.0, 4294967296.0, 1e10, -1e10, ep_ref.DENORM_MIN,
               -ep_ref.DENORM_MIN, ep_ref.FLT_MIN, 1.0000001, -0.9999999, 3.5, -3.5]


def cv_inputs(src):
    dt, lo, hi = CV_TYPES[src]
    rng = np.random.default_rng(zlib.crc32(('cv_' + src).encode()))
    if dt == 'f4':
        edges = np.array(FLOAT_EDGES, F32)
        r = np.concatenate([
            rng.uniform(-70000, 70000, 300).astype(F32),
            (rng.integers(-300, 300, 200) + rng.choice([0.0, 0.5, 0.25, 0.75], 200)).astype(F32),
            rng.uniform(-5e9, 5e9, 150).astype(F32),
            rng.integers(0, 2 ** 32, N - len(edges) - 650, dtype=np.uint64).astype(
                np.uint32).view(F32)])
        return np.concatenate([edges, r])
    edges = np.array([e for e in INT_EDGES if lo <= e <= hi], np.int64)
    r = np.concatenate([rng.integers(lo, hi, N // 2, endpoint=True),
                        rng.integers(max(lo, -70000), min(hi, 70000), N - N // 2 - len(edges),
                                     endpoint=True)])
    return np.concatenate([edges, r]).astype(dt)


def round_int(v, mode):
    """v (a Fraction) to an integer in mode."""
    fl = math.floor(v)
    rem = v - fl
    if mode == 'rtz':
        return int(fl) if v >= 0 or rem == 0 else int(fl) + 1
    if mode == 'rtn':
        return int(fl)
    if mode == 'rtp':
        return int(fl) + (rem > 0)
    return int(fl) + (rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl % 2 == 1))


def cv_int_want(v, dst, sat, mode, src_float):
    """The result convert_<dst>[_sat][_mode](v) must have, or None where it's
    implementation-defined (6.2.3.3)."""
    _, lo, hi = CV_TYPES[dst]
    if src_float and math.isnan(v):
        return None  # 6.2.3.3: _sat "should" make it 0, without it undefined
    if src_float and math.isinf(v):
        return (hi if v > 0 else lo) if sat else None
    n = round_int(Fraction(v), mode) if src_float else int(v)
    if lo <= n <= hi:
        return n
    if sat:
        return hi if n > hi else lo
    if src_float or lo < 0:
        # out of range: from float implementation-defined (6.2.3.3), to a
        # signed type implementation-defined (C99 6.3.1.3)
        return None
    return n % (hi + 1)  # C99 6.3.1.3: modulo for unsigned


def cv_check(src, dst):
    src_float = src == 'float'

    def check(ins, ref, dev):
        x = ins[0].reshape(-1)
        rows = []
        if dst != 'float':
            out = dev[1].reshape(-1, 10)
            nan_notes = set()
            for i, v in enumerate(x):
                v = py(v)
                for k, (sat, mode) in enumerate(INT_VARIANTS):
                    variants = [v]
                    if src_float and 0 < abs(v) < ep_ref.FLT_MIN:
                        variants += [0.0]  # 7.5.3 (3), sign immaterial here
                    wants = [cv_int_want(a, dst, sat, mode, src_float) for a in variants]
                    got = int(out[i, k])
                    ok = None in wants or got in wants
                    if sat and math.isnan(v) and got != 0:
                        nan_notes.add(got)
                    name = 'convert_%s%s%s' % (dst, '_sat' if sat else '',
                                               '' if k < 2 else '_' + mode)
                    rows.append(((name, v), got, None if ok else 'want %d' % wants[0]))
            if nan_notes:
                print('    convert_%s_sat(NaN) = %s (6.2.3.3: "should" be 0)' % (
                    dst, ', '.join(map(str, sorted(nan_notes)))))
            return collect('convert_%s from %s' % (dst, src), rows)
        out = dev[1].reshape(-1, 5)
        tally = {'rte': 0, 'rtz': 0}
        for i, v in enumerate(x):
            v = py(v)
            for k, mode in enumerate(FLOAT_VARIANTS):
                got = py(out[i, k])
                if src_float:
                    ok = (ep_ref.same(got, v) or (math.isnan(v) and math.isnan(got)) or
                          (0 < abs(v) < ep_ref.FLT_MIN and got == 0))
                    want = v
                elif v == 0:
                    ok, want = ep_ref.same(got, 0.0), 0.0
                else:
                    modes = (mode,) if mode else ('rtz', 'rte')
                    wants = [ep_ref.rnd(Fraction(int(v)), m) for m in modes]
                    ok, want = got in wants, wants[0]
                    if mode is None and ok and wants[0] != wants[1]:
                        tally['rtz' if got == wants[0] else 'rte'] += 1
                name = 'convert_float%s' % ('_' + mode if mode else '')
                rows.append(((name, v), got, None if ok else 'want %r' % want))
        if tally['rte'] or tally['rtz']:
            print('    convert_float(%s) rounds as %s (%d rtz, %d rte where they differ)' % (
                src, 'rtz' if tally['rtz'] >= tally['rte'] else 'rte', tally['rtz'], tally['rte']))
        if tally['rte'] and tally['rtz']:
            rows.append((('convert_float', src), None, 'the default rounding mode is mixed'))
        return collect('convert_float from %s' % src, rows)
    return check


def cv_case(src, dst):
    per = 5 if dst == 'float' else 10
    return Case('ep_cv_%s_%s' % (dst, src), K + 'ep_convert.cl', 'cv_%s_%s' % (dst, src), (N,),
                [Buf(CV_TYPES[src][0], N, values=cv_inputs(src)),
                 Buf(CV_TYPES[dst][0], per * N, 'zeros')], check=cv_check(src, dst),
                skip_compare=(1,))


CV_N = 64


def cv_vectors_inputs():
    rng = np.random.default_rng(33)
    x = (rng.integers(-3000, 3000, 33 * CV_N) + rng.choice([0.0, 0.5, 0.25, -0.5], 33 * CV_N))
    x = x.astype(F32)
    x[:9] = [2.5, -2.5, 3e9, np.nan, -0.5, 1.5, -1.5, 0.25, 2147483520.0]
    k = rng.integers(-2 ** 31, 2 ** 31, 33 * CV_N, dtype=np.int64).astype(np.int32)
    k[17:33] = [0, 255, 256, -1, 1000, -1000, 128, 127, 2 ** 31 - 1, -2 ** 31, 3, 4, 5, 6, 7, 8]
    return x, k


def cv_vectors_check(ins, ref, dev):
    x = ins[0].reshape(-1, 33)
    k = ins[1].reshape(-1, 33)
    y, f, u = dev[2].reshape(-1, 33), dev[3].reshape(-1, 33), dev[4].reshape(-1, 33)
    rows = []

    def want_int(v, dst, sat, mode):
        return cv_int_want(py(v), dst, sat, mode, True)
    for i in range(len(x)):
        for j, (sat, mode, name) in enumerate([(True, 'rte', 'convert_int2_sat_rte')] * 2 +
                                              [(False, 'rtz', 'convert_int3')] * 3 +
                                              [(False, 'rtp', 'convert_int4_rtp')] * 4):
            w = want_int(x[i, j], 'int', sat, mode)
            rows.append(((name, py(x[i, j])), int(y[i, j]),
                         None if w is None or w == int(y[i, j]) else 'want %d' % w))
        for j in range(9, 17):
            v = int(k[i, j])
            w = [ep_ref.rnd(Fraction(v), m) for m in ('rtz', 'rte')] if v else [0.0]
            rows.append((('convert_float8', v), py(f[i, j]),
                         None if py(f[i, j]) in w else 'want %r' % w[0]))
            w = ep_ref.rnd(Fraction(v), 'rtn') if v else 0.0
            rows.append((('convert_float8_rtn', v), py(f[i, j + 16]),
                         None if py(f[i, j + 16]) == w else 'want %r' % w))
        for j in range(17, 33):
            w = min(max(int(k[i, j]), 0), 255)
            rows.append((('convert_uchar16_sat', int(k[i, j])), int(u[i, j]),
                         None if w == int(u[i, j]) else 'want %d' % w))
    return collect('vector conversions', rows)


def as_types_check(ins, ref, dev):
    """6.2.4: the bits are kept, whatever they are (NaNs too)."""
    v = ins[0].reshape(-1, 4)
    y = dev[1].reshape(-1, 4, 4)
    want = np.stack([v, v, v[:, [0, 1, 0, 1]], v[:, [2, 3, 0, 3]]], axis=1)
    bad = np.nonzero((y != want).any(axis=2))
    errs = ['as_type: item %d row %d: %s, want %s' % (i, r, list(map(hex, y[i, r])),
                                                      list(map(hex, want[i, r])))
            for i, r in zip(*bad)][:6]
    if len(bad[0]):
        errs.append('as_type: %d of %d rows wrong' % (len(bad[0]), y.shape[0] * 4))
    return errs


_vx, _vk = cv_vectors_inputs()
_bits = np.random.default_rng(64).integers(0, 2 ** 32, 4 * 256, dtype=np.uint64).astype(np.uint32)
_bits[:8] = [0x7fc00000, 0x7f800001, 0xffffffff, 0x00000001, 0x80000000, 0x7f800000, 0xff800000,
             0x3f800000]
EP_CASES += [cv_case(s, d) for s in CV_TYPES for d in CV_TYPES]
EP_CASES += [
    Case('ep_cv_vectors', K + 'ep_convert.cl', 'cv_vectors', (CV_N,),
         [Buf('f4', 33 * CV_N, values=_vx), Buf('i4', 33 * CV_N, values=_vk),
          Buf('i4', 33 * CV_N, 'zeros'), Buf('f4', 33 * CV_N, 'zeros'),
          Buf('u1', 33 * CV_N, 'zeros')], check=cv_vectors_check, skip_compare=(2, 3, 4)),
    # the bits are compared with pocl's too
    Case('ep_as_types', K + 'ep_convert.cl', 'as_types', (256,),
         [Buf('u4', 4 * 256, values=_bits), Buf('u4', 16 * 256, 'zeros')], check=as_types_check),
]


# ---- section 6.11.3: integer functions, and 6.3's operators ----

INT_TYPES = [t for t in CV_TYPES if t != 'float']
INT_N = 512
SCALAR_OPS = ['abs', 'abs_diff', 'add_sat', 'hadd', 'rhadd', 'clamp', 'clz', 'mad_hi', 'mad_sat',
              'max', 'min', 'mul_hi', 'rotate', 'sub_sat', '+', '-', '*', '/', '%', '<<', '>>',
              '~x ^ y', '(x & y) | z', 'neg']
VECTOR_OPS = ['abs', 'add_sat', 'hadd', 'rhadd', 'clamp(x, s, t)', 'max(x, y.y)', 'min(x, y.z)',
              'mul_hi', 'rotate', 'sub_sat', 'mad_sat', '<<', '>>', 'clz', 'abs_diff', 'mad_hi']


def int_info(t):
    _, lo, hi = CV_TYPES[t]
    return (hi - lo).bit_length(), lo < 0, lo, hi


def to_type(v, t):
    """v in t, as C99 6.3.1.3 converts (modulo; the signed case is the
    implementation's, two's complement here)"""
    bits, signed, _, _ = int_info(t)
    v &= (1 << bits) - 1
    return v - (1 << bits) if signed and v >> (bits - 1) else v


def unsigned_of(t):
    return t if t.startswith('u') else 'u' + t


def int_ref(op, t, x, y, z, promote):
    """op on t-typed x, y, z: the value, before the (uint) or convert_uint.
    promote: the scalar kernel, where C99's integer promotions make the
    operators' arithmetic int (6.3: not for vectors)."""
    bits, signed, lo, hi = int_info(t)
    sat = lambda v: max(lo, min(hi, v))  # noqa: E731
    ar = (lambda v: to_type(v, 'int' if bits < 32 else t)) if promote else (lambda v: to_type(v, t))
    pat = lambda v: v & ((1 << bits) - 1)  # noqa: E731
    if op == 'abs':
        return abs(x)
    if op == 'abs_diff':
        return abs(x - y)
    if op == 'add_sat':
        return sat(x + y)
    if op == 'sub_sat':
        return sat(x - y)
    if op == 'hadd':
        return (x + y) >> 1
    if op == 'rhadd':
        return (x + y + 1) >> 1
    if op == 'clz':
        return bits - pat(x).bit_length()
    if op == 'mul_hi':
        return to_type((x * y) >> bits, t)
    if op == 'mad_hi':
        return to_type(((x * y) >> bits) + z, t)
    if op == 'mad_sat':
        return sat(x * y + z)
    if op == 'rotate':
        k = pat(y) % bits
        return to_type((pat(x) << k) | (pat(x) >> (bits - k)), t)
    if op == '+':
        return ar(x + y)
    if op == '-':
        return ar(x - y)
    if op == '*':
        return ar(x * y)
    if op in ('/', '%'):
        d = 1 if y == 0 else y
        if d == to_type(-1, t) and x == to_type(1 << (bits - 1), t):
            d = 1
        q = abs(x) // abs(d) * (1 if (x < 0) == (d < 0) else -1)
        return ar(q) if op == '/' else ar(x - q * d)
    if op in ('<<', '>>'):
        # the scalar kernel masks the count, a vector's is taken modulo (6.3.j)
        k = pat(y) % bits
        return ar(x << k) if op == '<<' else ar(x >> k)
    if op == '~x ^ y':
        return ar(~x ^ y)
    if op == '(x & y) | z':
        return ar((x & y) | z)
    if op == 'neg':
        return ar(-x)
    if op == 'upsample':
        return x * (1 << bits) + pat(y)
    if op in ('mul24', 'mad24'):
        p = to_type(x * y, t)
        return p if op == 'mul24' else to_type(p + z, t)
    raise ValueError(op)


def int_inputs(t, n, salt):
    _, _, lo, hi = int_info(t)
    rng = np.random.default_rng(zlib.crc32(('int_%s_%s' % (t, salt)).encode()))
    edges = [e for e in (0, 1, -1, 2, -2, 7, 8, lo, lo + 1, hi, hi - 1, hi // 2, lo // 2)
             if lo <= e <= hi]
    out = []
    for k in range(3):
        # every pair of edge values among the first items, then random
        e = np.array([edges[(i // len(edges) ** k) % len(edges)] for i in range(len(edges) ** 2)]
                     if k < 2 else [edges[i % len(edges)] for i in range(len(edges) ** 2)], np.int64)
        r = np.concatenate([rng.integers(lo, hi, (n - len(e)) // 2, endpoint=True),
                            rng.integers(max(lo, -300), min(hi, 300), n - len(e) - (n - len(e)) // 2,
                                         endpoint=True)])
        out.append(np.concatenate([e, r]).astype(CV_TYPES[t][0]))
    return out


def int_scalar_check(t):
    bits = int_info(t)[0]
    names = SCALAR_OPS + (['upsample(x, y)', 'upsample(y, z)'] if bits < 32 else ['mul24', 'mad24'])

    def check(ins, ref, dev):
        a, b, c = (ins[k].reshape(-1) for k in range(3))
        o = dev[3].reshape(-1, 26)
        errs = []
        for j, name in enumerate(names):
            rows = []
            for i in range(len(a)):
                x, y, z = int(a[i]), int(b[i]), int(c[i])
                lo, hi = min(y, z), max(y, z)
                if name == 'clamp':
                    want = max(lo, min(hi, x))
                elif name in ('max', 'min'):
                    want = max(x, y) if name == 'max' else min(x, y)
                elif name == 'upsample(x, y)':
                    want = int_ref('upsample', t, x, y, 0, True)
                elif name == 'upsample(y, z)':
                    want = int_ref('upsample', t, y, z, 0, True)
                elif name in ('mul24', 'mad24'):
                    x24, y24 = to_type(x << 8, t) >> 8, to_type(y << 8, t) >> 8
                    want = int_ref(name, t, x24, y24, z, True)
                else:
                    want = int_ref(name, t, x, y, z, True)
                got = int(o[i, j])
                rows.append(((x, y, z), got, None if got == want & 0xffffffff
                             else 'want %d' % (want & 0xffffffff)))
            errs += collect('%s %s' % (t, name), rows)
        return errs
    return check


def int_vector_check(t):
    def check(ins, ref, dev):
        a, b, c = (ins[k].reshape(-1, 4) for k in range(3))
        o = dev[3].reshape(-1, 16, 4)
        errs = []
        for j, name in enumerate(VECTOR_OPS):
            rows = []
            for i in range(len(a)):
                s, u = min(int(b[i, 0]), int(c[i, 0])), max(int(b[i, 0]), int(c[i, 0]))
                for e in range(4):
                    x, y, z = int(a[i, e]), int(b[i, e]), int(c[i, e])
                    if name == 'clamp(x, s, t)':
                        want = max(s, min(u, x))
                    elif name == 'max(x, y.y)':
                        want = max(x, int(b[i, 1]))
                    elif name == 'min(x, y.z)':
                        want = min(x, int(b[i, 2]))
                    else:
                        want = int_ref(name, t, x, y, z, False)
                    got = int(o[i, j, e])
                    rows.append(((x, y, z), got, None if got == want & 0xffffffff
                                 else 'want %d' % (want & 0xffffffff)))
            errs += collect('%s4 %s' % (t, name), rows)
        return errs
    return check


def int_cases():
    cases = []
    for t in INT_TYPES:
        dt = CV_TYPES[t][0]
        ins = int_inputs(t, INT_N, 's')
        cases.append(Case('ep_i_' + t, K + 'ep_integer.cl', 'i_' + t, (INT_N,),
                          [Buf(dt, INT_N, values=v) for v in ins] + [Buf('u4', 26 * INT_N, 'zeros')],
                          check=int_scalar_check(t)))
        ins = int_inputs(t, INT_N, 'v')
        cases.append(Case('ep_v_' + t, K + 'ep_integer.cl', 'v_' + t, (INT_N // 4,),
                          [Buf((dt, 4), INT_N // 4, values=v) for v in ins] +
                          [Buf(('u4', 4), 16 * INT_N // 4, 'zeros')],
                          check=int_vector_check(t)))
    return cases


EP_CASES += int_cases()


# ---- section 6.2.4.2: as_type between every pair of types of one size ----

AS_GROUPS = {2: 'ushort short uchar2 char2'.split(),
             4: 'uint int float uchar4 char4 ushort2 short2'.split(),
             8: 'uint2 int2 float2 uchar8 char8 ushort4 short4'.split(),
             16: 'uint4 int4 float4 uchar16 char16 ushort8 short8'.split(),
             32: 'uint8 int8 float8 ushort16 short16'.split()}
AS_N = 64


def as_check(size, types):
    def check(ins, ref, dev):
        x = ins[0].view(np.uint8).reshape(AS_N, size)
        o = dev[1].reshape(AS_N, len(types), size)
        rows = []
        for i in range(AS_N):
            for k, d in enumerate(types):
                got = o[i, k].tobytes()
                rows.append(((d, x[i].tobytes().hex()), got.hex(),
                             None if got == x[i].tobytes() else 'want the same bytes'))
        return collect('as_type', rows)
    return check


def as_cases():
    rng = np.random.default_rng(6242)
    cases = []
    for size, types in AS_GROUPS.items():
        words = rng.integers(0, 2 ** 32, AS_N * size // 4, dtype=np.uint64).astype(np.uint32)
        # NaNs (quiet, signalling, with payloads), infinities, -0 and a denormal
        words[:8] = [0x7fc00000, 0x7f800001, 0xffc12345, 0x7fbfffff, 0x7f800000, 0x80000000,
                     0x00000001, 0x01020304]
        for s in types:
            base = s.rstrip('0123456789')
            n = int(s[len(base):] or 1)
            dt = (CV_TYPES[base][0], n) if n > 1 else CV_TYPES[base][0]
            vals = words.view(np.dtype(CV_TYPES[base][0]))
            cases.append(Case('ep_as_' + s, K + 'ep_astype.cl', 'a_' + s, (AS_N,),
                              [Buf(dt, AS_N, values=vals), Buf('u1', AS_N * size * len(types), 'zeros')],
                              check=as_check(size, types)))
    return cases


EP_CASES += as_cases()


# ---- sections 6.11.4-6.11.6: common, geometric and relational functions ----
# The spec gives these no ulp bounds (table 10.1 has none; 6.11.5 gives
# fast_normalize 8192 ulp): those it defines exactly (clamp, max, min, step,
# sign, the relational functions, select, bitselect) are checked exactly;
# mix and cross must be their formula with each operation correctly rounded
# in one of the device's modes, or contracted to fma (6.11.4 note 49, 6.11.5
# note 50); the rest within bounds from the operations' (table 10.1), and
# their worst error is printed.

CG_N = 1024
mp = ep_ref.mp
RND_MODES = ('rte', 'rtz')


def r32(v, mode):
    """the exact v rounded to float in mode (0 stays 0)"""
    v = Fraction(v)
    return 0.0 if v == 0 else ep_ref.rnd(v, mode)


def mix_values(x, y, t):
    """x + (y - x) * t, each operation rounded in either mode, or fused"""
    out = set()
    for m1 in RND_MODES:
        d = r32(Fraction(y) - Fraction(x), m1)
        for m2 in RND_MODES:
            out.add(r32(Fraction(x) + Fraction(d) * Fraction(t), m2))  # fma
            for m3 in RND_MODES:
                out.add(r32(Fraction(x) + Fraction(r32(Fraction(d) * Fraction(t), m2)), m3))
    return out


def diff_products(a, b, c, d):
    """a * b - c * d, each operation rounded in either mode, or one fused"""
    out = set()
    A, B, C, D = map(Fraction, (a, b, c, d))
    for m1 in RND_MODES:
        p, q = r32(A * B, m1), r32(C * D, m1)
        for m2 in RND_MODES:
            out.add(r32(Fraction(p) - Fraction(q), m2))
            out.add(r32(A * B - Fraction(q), m2))
            out.add(r32(Fraction(p) - C * D, m2))
    return out


def near(dev, exact, tol):
    """|dev - exact| <= tol (exact a Fraction or mpf)"""
    return math.isfinite(dev) and abs(mp.mpf(dev) - exact) <= tol


def exact_mpf(v):
    return mp.mpf(v.numerator) / v.denominator if isinstance(v, Fraction) else mp.mpf(v)


EPS = mp.ldexp(1, -23)


def smoothstep_ref(e0, e1, x):
    t = (Fraction(x) - Fraction(e0)) / (Fraction(e1) - Fraction(e0))
    t = min(max(t, Fraction(0)), Fraction(1))
    return t, t * t * (3 - 2 * t)


def common_inputs(salt, n):
    rng = np.random.default_rng(zlib.crc32(('common' + salt).encode()))
    edges = np.array([0.0, -0.0, 1.0, -1.0, 0.5, 2.5, -2.5, 1e-3, -1e30, 1e30, 3.0, 100.0],
                     np.float32)
    def col(k):
        r = np.concatenate([rng.uniform(-100, 100, n // 2), rng.uniform(-1, 1, n // 4),
                            np.ldexp(rng.uniform(-1, 1, n), rng.integers(-60, 60, n))])[:n - len(edges)]
        return np.concatenate([np.roll(edges, k), r]).astype(np.float32)
    t = np.concatenate([[0.0, 1.0, 0.5, 0.25], rng.uniform(0, 1, n - 4)]).astype(np.float32)
    return col(0), col(3), col(7), t


def common_value(name, x, y, lo, hi, t):
    """(accepted values (a set), or None to check against (exact, tol))"""
    if name == 'clamp':
        return {min(max(x, lo), hi)}, None
    if name == 'max':
        return {y if x < y else x}, None
    if name == 'min':
        return {y if y < x else x}, None
    if name == 'step':
        return {0.0 if x < y else 1.0}, None
    if name == 'sign':
        return {1.0 if x > 0 else -1.0 if x < 0 else x}, None
    if name == 'mix':
        # footnote 49 allows mad, of no set accuracy: 4 ulp of the operands'
        e = mp.mpf(x) + (mp.mpf(y) - mp.mpf(x)) * mp.mpf(t)
        mag = max(abs(mp.mpf(x)), abs(mp.mpf(y) - mp.mpf(x)) * mp.mpf(t))
        return None, (e, 4 * EPS * mag + mp.ldexp(1, -126))
    if name in ('degrees', 'radians'):
        e = mp.mpf(x) * (180 / mp.pi if name == 'degrees' else mp.pi / 180)
        # one multiply by the rounded constant: 2 ulp
        return None, (e, 2 * ep_ref.ulp(e) if e else 0)
    if name == 'smoothstep':
        if not lo < hi:
            return 'skip', None
        tt, v = smoothstep_ref(lo, hi, x)
        e = exact_mpf(v)
        # t's relative error: two subtractions (1 ulp each), the divide (3),
        # its rounding; the polynomial's slope is at most 1.5 and its three
        # operations add an ulp of the result each
        tol = 1.5 * 8 * EPS * exact_mpf(tt) + 4 * (ep_ref.ulp(e) if e else 0) + mp.ldexp(1, -126)
        return None, (e, tol)
    raise ValueError(name)


COMMON_SCALAR = ['clamp', 'degrees', 'radians', 'max', 'min', 'mix', 'step', 'smoothstep', 'sign']


def zero_eq(a, b):
    """equal, either zero for either (min, max and clamp compare zeros equal;
    section 10 item 6)"""
    return a == b or (math.isnan(a) and math.isnan(b))


def common_rows(name, x, y, z, t, got, rows, worst):
    lo, hi = min(y, z), max(y, z)
    if name in ('clamp', 'smoothstep'):
        y = None
    vals, tol = common_value(name, x, y, lo, hi, t)
    if vals == 'skip':
        return
    if vals is not None:
        ok = any(zero_eq(got, v) if name in ('clamp', 'max', 'min') else
                 ep_ref.same(got, v) or (v == 0 and got == 0 and math.copysign(1, got) > 0)
                 for v in vals)
        rows.append(((x, y, z, t), got, None if ok else 'want %r' % sorted(vals)))
        return
    e, tl = tol
    ok = near(got, e, tl)
    if math.isfinite(got) and e:
        worst[name] = max(worst.get(name, 0), float(abs(mp.mpf(got) - e) / ep_ref.ulp(e)))
    rows.append(((x, y, z, t), got, None if ok else 'want %s +- %s' % (mp.nstr(e, 9), mp.nstr(tl, 3))))


def print_worst(worst):
    for k, v in sorted(worst.items()):
        print('    %s: at most %.1f ulp' % (k, v))


def common_scalar_check(ins, ref, dev):
    a, b, c, d = (ins[k].reshape(-1) for k in range(4))
    o = dev[4].reshape(-1, 9)
    errs, worst = [], {}
    for j, name in enumerate(COMMON_SCALAR):
        rows = []
        for i in range(len(a)):
            common_rows(name, py(a[i]), py(b[i]), py(c[i]), py(d[i]), py(o[i, j]), rows, worst)
        errs += collect(name, rows)
    print_worst(worst)
    return errs


COMMON_VECTOR = COMMON_SCALAR + ['clamp(x, s, u)', 'max(x, s)', 'min(x, s)', 'mix(x, y, w)',
                                 'step(s, x)', 'smoothstep(s, u, x)']


def common_vector_check(ins, ref, dev):
    a, b, c, d = (ins[k].reshape(-1, 4) for k in range(4))
    o = dev[4].reshape(-1, 15, 4)
    errs, worst = [], {}
    for j, name in enumerate(COMMON_VECTOR):
        rows = []
        base = name.split('(')[0]
        for i in range(len(a)):
            s, u, w = min(py(b[i, 0]), py(c[i, 0])), max(py(b[i, 0]), py(c[i, 0])), py(d[i, 0])
            for e in range(4):
                x, y, z, t = py(a[i, e]), py(b[i, e]), py(c[i, e]), py(d[i, e])
                if name == 'clamp(x, s, u)' or name == 'smoothstep(s, u, x)':
                    y, z = s, u
                elif name in ('max(x, s)', 'min(x, s)', 'step(s, x)'):
                    y = s
                elif name == 'mix(x, y, w)':
                    t = w
                common_rows(base, x, y, z, t, py(o[i, j, e]), rows, worst)
        errs += collect(name, rows)
    print_worst(worst)
    return errs


def geo_inputs():
    rng = np.random.default_rng(6115)
    p = np.ldexp(rng.uniform(-1, 1, (512, 4)), rng.integers(-20, 20, (512, 4)))
    q = np.ldexp(rng.uniform(-1, 1, (512, 4)), rng.integers(-20, 20, (512, 4)))
    p[:4] = [[0, 0, 0, 0], [1, 0, 0, 0], [0, -3, 4, 0], [1e-30, 1e-30, 0, 1e-30]]
    q[:4] = [[0, 0, 0, 0], [0, 1, 0, 0], [3, 0, 0, 4], [2, 3, 4, 5]]
    return p.astype(np.float32), q.astype(np.float32)


def geo_check(ins, ref, dev):
    P, Q = ins[0].reshape(-1, 4), ins[1].reshape(-1, 4)
    o = dev[2].reshape(-1, 48)
    rows = {k: [] for k in ('dot', 'distance', 'length', 'fast_distance', 'fast_length',
                            'normalize', 'fast_normalize', 'cross')}
    worst = {}

    def chk(name, args, got, e, tol, ref_ulp=None):
        if e is not None and math.isfinite(got) and ref_ulp:
            worst[name] = max(worst.get(name, 0), float(abs(mp.mpf(got) - e) / ref_ulp))
        rows[name].append((args, got, None if near(got, e, tol) else
                           'want %s +- %s' % (mp.nstr(e, 9), mp.nstr(tol, 3))))
    for i in range(len(P)):
        p, q = [py(v) for v in P[i]], [py(v) for v in Q[i]]
        base = 0
        for n in (1, 2, 3, 4):
            pp, qq = p[:n], q[:n]
            dotv = sum(mp.mpf(a) * mp.mpf(b) for a, b in zip(pp, qq))
            mag = sum(abs(mp.mpf(a) * mp.mpf(b)) for a, b in zip(pp, qq))
            # n products and n - 1 sums, each an ulp of the magnitudes'
            chk('dot', (n, pp, qq), py(o[i, base]), dotv, (n + 1) * EPS * mag + mp.ldexp(1, -126),
                ep_ref.ulp(dotv) if dotv else None)
            for name, vec, k, extra in (('distance', [mp.mpf(a) - mp.mpf(b) for a, b in zip(pp, qq)], 1, 1),
                                        ('length', [mp.mpf(a) for a in pp], 2, 0)):
                sq = sum(v * v for v in vec)
                if not ep_ref.FLT_MIN <= sq <= ep_ref.FLT_MAX:
                    continue  # 6.11.5 doesn't ask to avoid the squares' underflow or overflow
                L = mp.sqrt(sq)
                # the squares and sums (n + 1 ulp, halved by the root), the
                # root (4), the differences (1)
                tol = ((n + 1) / 2 + 4 + extra) * EPS * L + mp.ldexp(1, -126)
                chk(name, (n, pp, qq), py(o[i, base + k]), L, tol, ep_ref.ulp(L) if L else None)
                ftol = (8192 + n + 2) * EPS * L + mp.ldexp(1, -63)
                chk('fast_' + name, (n, pp, qq), py(o[i, base + k + 2]), L, ftol,
                        ep_ref.ulp(L) if L else None)
            base += 5
        nb = [20, 21, 23, 26]
        for n, at in zip((1, 2, 3, 4), nb):
            pp = [mp.mpf(a) for a in p[:n]]
            s = sum(v * v for v in pp)
            for name, off, ulps in (('normalize', 0, n + 10), ('fast_normalize', 10, 8192 + n + 2)):
                if s > ep_ref.FLT_MAX or (name == 'normalize' and 0 < s < ep_ref.FLT_MIN):
                    continue  # undefined (6.11.5's exceptions for fast_normalize)
                for k in range(n):
                    got = py(o[i, at + off + k])
                    if s == 0:
                        rows[name].append(((n, p[:n]), got, None if name == 'normalize' or got == p[k]
                                           else 'want %r (p)' % p[k]))
                        continue
                    if s < ep_ref.FLT_MIN and name == 'fast_normalize' and got == p[k]:
                        rows[name].append(((n, p[:n]), got, None))
                        continue
                    e = pp[k] / mp.sqrt(s)
                    chk(name, (n, p[:n]), got, e, ulps * EPS * max(abs(e), mp.ldexp(1, -24)) * 1,
                        ep_ref.ulp(e) if e else None)
        c4 = [py(v) for v in o[i, 40:44]]
        c3 = [py(v) for v in o[i, 44:47]]
        for got_v, n in ((c4, 4), (c3, 3)):
            for k, (a, b, c, d) in enumerate([(p[1], q[2], p[2], q[1]), (p[2], q[0], p[0], q[2]),
                                              (p[0], q[1], p[1], q[0])]):
                # footnote 50 allows mad, of no set accuracy: 4 ulp of the products'
                ab, cd = mp.mpf(a) * mp.mpf(b), mp.mpf(c) * mp.mpf(d)
                e = ab - cd
                chk('cross', (n, p[:3], q[:3]), got_v[k], e,
                    4 * EPS * max(abs(ab), abs(cd)) + mp.ldexp(1, -126), ep_ref.ulp(e) if e else None)
            if n == 4:
                rows['cross'].append(((4, p, q), c4[3], None if ep_ref.same(c4[3], 0.0) else 'want 0.0 in w'))
    errs = []
    for k, r in rows.items():
        errs += collect(k, r)
    print_worst(worst)
    return errs


REL_NAMES = ['isequal', 'isnotequal', 'isgreater', 'isgreaterequal', 'isless', 'islessequal',
             'islessgreater', 'isfinite', 'isinf', 'isnan', 'isnormal', 'isordered', 'isunordered',
             'signbit', '==', '!=', '<', '>', '<=', '>=']


def rel_value(name, x, y):
    nan = math.isnan(x) or math.isnan(y)
    return {'isequal': x == y, 'isnotequal': x != y, 'isgreater': x > y, 'isgreaterequal': x >= y,
            'isless': x < y, 'islessequal': x <= y, 'islessgreater': (x < y) or (x > y),
            'isfinite': math.isfinite(x), 'isinf': math.isinf(x), 'isnan': math.isnan(x),
            'isnormal': math.isfinite(x) and abs(x) >= ep_ref.FLT_MIN,
            'isordered': not nan, 'isunordered': nan, 'signbit': math.copysign(1, x) < 0,
            '==': x == y, '!=': x != y, '<': x < y, '>': x > y, '<=': x <= y, '>=': x >= y}[name]


REL_SPECIALS = [math.nan, math.inf, -math.inf, 0.0, -0.0, 1.0, -1.0, 2.0, ep_ref.FLT_MAX,
                -ep_ref.FLT_MAX, ep_ref.FLT_MIN, -ep_ref.FLT_MIN, ep_ref.DENORM_MIN, -ep_ref.DENORM_MIN,
                1.5, -1.5]


def rel_inputs(n):
    rng = np.random.default_rng(6116)
    pairs = [(a, b) for a in REL_SPECIALS for b in REL_SPECIALS]
    x = np.array([a for a, _ in pairs] + list(rng.uniform(-3, 3, n - len(pairs))), np.float32)
    y = np.array([b for _, b in pairs] + list(np.where(rng.random(n - len(pairs)) < 0.3,
                                                       x[len(pairs):], rng.uniform(-3, 3, n - len(pairs)))),
                 np.float32)
    return x, y


def rel_check(vector):
    def check(ins, ref, dev):
        x, y = ins[0].reshape(-1), ins[1].reshape(-1)
        o = dev[2].reshape(-1, 20, 4) if vector else dev[2].reshape(-1, 20)
        true = -1 if vector else 1
        errs = []
        for j, name in enumerate(REL_NAMES):
            rows = []
            for i in range(len(x)):
                a, b = py(x[i]), py(y[i])
                got = int(o[i // 4, j, i % 4]) if vector else int(o[i, j])
                # a sub-normal operand may be flushed to either zero (7.5.3)
                wants = {true if rel_value(name, u, v) else 0 for u, v in ep_ref.ftz_variants((a, b))}
                rows.append(((a, b), got, None if got in wants else 'want %s' % sorted(wants)))
            errs += collect(('%s (vector)' if vector else '%s') % name, rows)
        return errs
    return check


def branch_check(ins, ref, dev):
    x, y = ins[0].reshape(-1), ins[1].reshape(-1)
    o = dev[2].reshape(-1, 6)
    errs = []
    for j, name in enumerate(REL_NAMES[14:]):
        rows = []
        for i in range(len(x)):
            a, b = py(x[i]), py(y[i])
            wants = {1 if rel_value(name, u, v) else 0 for u, v in ep_ref.ftz_variants((a, b))}
            rows.append(((a, b), int(o[i, j]), None if int(o[i, j]) in wants else 'want %s' % sorted(wants)))
        errs += collect('if (x %s y)' % name, rows)
    return errs


NEGATED = [('!(x < y)', lambda a, b: not a < b), ('!(x <= y)', lambda a, b: not a <= b),
           ('!(x > y)', lambda a, b: not a > b), ('!(x >= y)', lambda a, b: not a >= b),
           ('x < y || x > y', lambda a, b: a < b or a > b),
           ('!(x < y || x > y)', lambda a, b: not (a < b or a > b))]


def negated_check(ins, ref, dev):
    x, y = ins[0].reshape(-1), ins[1].reshape(-1)
    o = dev[2].reshape(-1, 6)
    errs = []
    for j, (name, f) in enumerate(NEGATED):
        rows = []
        for i in range(len(x)):
            a, b = py(x[i]), py(y[i])
            wants = {int(f(u, v)) for u, v in ep_ref.ftz_variants((a, b))}
            rows.append(((a, b), int(o[i, j]), None if int(o[i, j]) in wants else 'want %s' % sorted(wants)))
        errs += collect(name, rows)
    return errs


def sel_inputs(n):
    rng = np.random.default_rng(6117)
    x = rng.integers(0, 2 ** 32, (n, 4), dtype=np.uint64).astype(np.uint32)
    y = rng.integers(0, 2 ** 32, (n, 4), dtype=np.uint64).astype(np.uint32)
    m = rng.integers(0, 2 ** 32, (n, 4), dtype=np.uint64).astype(np.uint32)
    m[: n // 4] |= 0x80808080          # every top bit set
    m[n // 4: n // 2] &= 0x7f7f7f7f    # none
    m[n // 2: n // 2 + 8] = 0
    return x, y, m


def sel_check(ins, ref, dev):
    X, Y, Mk = (ins[k].reshape(-1, 4) for k in range(3))
    o = dev[3].reshape(-1, 40)
    rows = []

    def top(words, width):
        """the top bits of each width-bit element of the uint words"""
        b = []
        for w in words:
            for k in range(32 // width):
                b.append((int(w) >> (width * k + width - 1)) & 1)
        return b

    def sel(xw, yw, mw, width, vector):
        out = []
        for a, b, c in zip(xw, yw, mw):
            v = 0
            for k in range(32 // width):
                sh, msk = width * k, (1 << width) - 1
                ce = (int(c) >> sh) & msk
                pick = (ce >> (width - 1)) & 1 if vector else ce != 0
                v |= (((int(b) if pick else int(a)) >> sh) & msk) << sh
            out.append(v)
        return out
    for i in range(len(X)):
        x, y, m = list(X[i]), list(Y[i]), list(Mk[i])
        bsel = [(int(a) & ~int(c) | int(b) & int(c)) & 0xffffffff for a, b, c in zip(x, y, m)]
        want = [int(any(top(m, 32))), int(all(top(m, 32))), int(any(top(m, 8))), int(all(top(m, 8))),
                int(any(top(m, 16))), int(all(top(m, 16))), top(m[:1], 32)[0], top(m[:1], 32)[0],
                int(any(top(m[:2], 32))), int(all(top(m[:3], 32)))]
        want += bsel + bsel + sel(x, y, m, 32, True) + sel(x, y, m, 32, True) + \
            sel(x, y, m, 8, True) + sel(x, y, m, 16, True)
        want += [int(y[0]) if m[0] else int(x[0]), int(y[1]) if m[1] else int(x[1]),
                 int(y[2]) if m[2] & 1 else int(x[2]), bsel[3],
                 int(y[0]) if m[2] & 0x80000000 else int(x[0]), 0]
        got = [int(v) for v in o[i]]
        for k, (g, w) in enumerate(zip(got, want)):
            rows.append(((i, k), g, None if g == w else 'want 0x%x' % w))
    return collect('any/all/select/bitselect', rows)


def common_cases():
    sx = common_inputs('s', CG_N)
    vx = common_inputs('v', CG_N)
    gp, gq = geo_inputs()
    rx, ry = rel_inputs(CG_N)
    sx_, sy_, sm_ = sel_inputs(256)
    return [
        Case('ep_c_float', K + 'ep_common.cl', 'c_float', (CG_N,),
             [Buf('f4', CG_N, values=v) for v in sx] + [Buf('f4', 9 * CG_N, 'zeros')],
             check=common_scalar_check, skip_compare=(4,)),
        Case('ep_c_float4', K + 'ep_common.cl', 'c_float4', (CG_N // 4,),
             [Buf(('f4', 4), CG_N // 4, values=v) for v in vx] + [Buf(('f4', 4), 15 * CG_N // 4, 'zeros')],
             check=common_vector_check, skip_compare=(4,)),
        Case('ep_g_float', K + 'ep_common.cl', 'g_float', (512,),
             [Buf(('f4', 4), 512, values=gp), Buf(('f4', 4), 512, values=gq), Buf('f4', 48 * 512, 'zeros')],
             check=geo_check, skip_compare=(2,)),
        Case('ep_r_float', K + 'ep_common.cl', 'r_float', (CG_N,),
             [Buf('f4', CG_N, values=rx), Buf('f4', CG_N, values=ry), Buf('i4', 20 * CG_N, 'zeros')],
             check=rel_check(False), skip_compare=(2,)),
        Case('ep_r_float4', K + 'ep_common.cl', 'r_float4', (CG_N // 4,),
             [Buf(('f4', 4), CG_N // 4, values=rx), Buf(('f4', 4), CG_N // 4, values=ry),
              Buf(('i4', 4), 20 * CG_N // 4, 'zeros')], check=rel_check(True), skip_compare=(2,)),
        Case('ep_r_branch', K + 'ep_common.cl', 'r_branch', (CG_N,),
             [Buf('f4', CG_N, values=rx), Buf('f4', CG_N, values=ry), Buf('i4', 6 * CG_N, 'zeros')],
             check=branch_check, skip_compare=(2,)),
        Case('ep_r_negated', K + 'ep_common.cl', 'r_negated', (CG_N,),
             [Buf('f4', CG_N, values=rx), Buf('f4', CG_N, values=ry), Buf('i4', 6 * CG_N, 'zeros')],
             check=negated_check, skip_compare=(2,)),
        Case('ep_r_select', K + 'ep_common.cl', 'r_select', (256,),
             [Buf(('u4', 4), 256, values=sx_), Buf(('u4', 4), 256, values=sy_),
              Buf(('u4', 4), 256, values=sm_), Buf('u4', 40 * 256, 'zeros')], check=sel_check),
    ]


EP_CASES += common_cases()


# ---- sections 6.11.7, 6.11.9, 6.11.10, 6.11.12 and 6.9 (ep_misc.cl) ----

HL_N = 1024   # items, 16 halves or floats each


def half_inputs():
    h = np.arange(0, 65536, 4, dtype=np.uint32)
    # every denormal, and the infinities, NaNs and largest
    h[:1024] = np.arange(1024)
    h[1024:1040] = [0x7c00, 0xfc00, 0x7e00, 0xfe00, 0x7c01, 0x7dff, 0x7fff, 0xffff, 0x7bff, 0xfbff,
                    0x0400, 0x8400, 0x3c00, 0xbc00, 0x8001, 0x83ff]
    return h.astype(np.uint16)


def h2f(h):
    return np.asarray(h, np.uint16).view(np.float16).astype(np.float32)


def half_load_check(ins, ref, dev):
    h = ins[0].reshape(HL_N, 16)
    c = ins[1].reshape(32, 16)
    got = dev[2].reshape(HL_N, 15, 16)
    names = ['vload_half', 'vload_half2', 'vload_half3', 'vload_half4', 'vload_half8',
             'vload_half16', 'vloada_half', 'vloada_half2', 'vloada_half3', 'vloada_half4',
             'vloada_half8', 'vloada_half16', 'vload_half16 __constant',
             'vload_half16 __private', 'vloada_half16 __local']
    errs = []
    for row, name in enumerate(names):
        src = c[np.arange(HL_N) & 31] if row == 12 else h
        want = h2f(src)
        d = got[:, row]
        if row == 8:  # a half3 every 4: the 4th untouched
            want = want.copy()
            want[:, 3::4] = 0
        wb, db = want.view(np.uint32), d.view(np.uint32)
        # section 10 item 7: a denormal half may load as 0
        denorm = ((src & 0x7c00) == 0) & ((src & 0x3ff) != 0)
        ok = (wb == db) | (np.isnan(want) & np.isnan(d)) | (denorm & ((db & 0x7fffffff) == 0))
        if row == 8:
            ok[:, 3::4] = db[:, 3::4] == 0
        rows = [((hex(int(src[i, k])),), float(d[i, k]), None if ok[i, k] else 'want %r' % float(want[i, k]))
                for i in range(HL_N) for k in range(16)]
        errs += collect(name, rows)
    return errs


def half_store_inputs():
    rng = np.random.default_rng(1170)
    x = [rng.uniform(-7e4, 7e4, 2048), rng.standard_normal(4096) * np.exp2(rng.uniform(-30, 17, 4096))]
    # each side of halves, and between them
    h = h2f(rng.integers(0, 0x7c00, 2048)).astype(np.float64)
    nxt = h2f(np.asarray(h, np.float32).astype(np.float16).view(np.uint16) + 1).astype(np.float64)
    x += [h, np.nextafter(h.astype(np.float32), np.float32(np.inf)),
          np.nextafter(h.astype(np.float32), np.float32(0)), (h + nxt) / 2]
    sp = [0.0, -0.0, np.inf, -np.inf, np.nan, ep_ref.DENORM_MIN, ep_ref.FLT_MIN, 65504.0, 65519.996,
          65520.0, 65536.0, ep_ref.FLT_MAX, 2.0 ** -24, 2.0 ** -25, 3 * 2.0 ** -26, 2.0 ** -14,
          float(np.nextafter(np.float32(2.0 ** -14), np.float32(0))), 1.0, 1.0 + 2.0 ** -11,
          1.0 + 3 * 2.0 ** -11, 2049.0, 2051.0]
    x.append(sp)
    x = np.concatenate([np.asarray(v, np.float64) for v in x]).astype(np.float32)
    x = np.concatenate([x, -x])
    rng.shuffle(x)
    out = np.zeros(HL_N * 16, np.float32)
    out[:len(sp) * 2] = np.concatenate([sp, -np.asarray(sp)])
    out[len(sp) * 2:] = x[:HL_N * 16 - len(sp) * 2]
    return out


def f2h_ref(x, mode):
    """numpy's round to nearest even, then a step to the mode's side"""
    with np.errstate(over='ignore', invalid='ignore'):
        h = x.astype(np.float16)
    b = h.view(np.uint16).astype(np.int64)
    hv = h.astype(np.float64)
    xv = x.astype(np.float64)
    mag, sign = b & 0x7fff, b & 0x8000
    fin = ~np.isnan(xv) & ~np.isinf(xv)
    away = fin & (np.abs(hv) > np.abs(xv))
    if mode == 'rtz':
        mag = np.where(away, mag - 1, mag)
    elif mode in ('rtp', 'rtn'):
        grow = fin & ((hv < xv) if mode == 'rtp' else (hv > xv))
        neg = np.signbit(x)
        step = np.where(neg, -1, 1) if mode == 'rtp' else np.where(neg, 1, -1)
        mag = np.where(grow, mag + step, mag)
    return (sign | mag).astype(np.uint16)


# h_store's rows: mode, and whether the 4th of every 4 is untouched
HS_ROWS = [('vstore_half', 'rte'), ('vstore_half_rte', 'rte'), ('vstore_half_rtz', 'rtz'),
           ('vstore_half_rtp', 'rtp'), ('vstore_half_rtn', 'rtn'), ('vstore_half2', 'rte'),
           ('vstore_half3', 'rte'), ('vstore_half4', 'rte'), ('vstore_half8', 'rte'),
           ('vstore_half16', 'rte'), ('vstore_half2_rtz', 'rtz'), ('vstore_half3_rtp', 'rtp'),
           ('vstore_half4_rtn', 'rtn'), ('vstore_half8_rte', 'rte'), ('vstore_half16_rtz', 'rtz'),
           ('vstorea_half', 'rte'), ('vstorea_half2_rtp', 'rtp'), ('vstorea_half3_rtn', 'rtn'),
           ('vstorea_half4', 'rte'), ('vstorea_half8_rtz', 'rtz'), ('vstorea_half16_rte', 'rte'),
           ('vstore_half16_rtn __local', 'rtn'), ('vstore_half8 vstorea_half8_rtp __private', None)]


def half_store_check(ins, ref, dev):
    x = ins[0].reshape(HL_N, 16)
    got = dev[1].reshape(HL_N, 23, 16)
    want = {m: f2h_ref(x, m) for m in ('rte', 'rtz', 'rtp', 'rtn')}
    errs = []
    for row, (name, mode) in enumerate(HS_ROWS):
        w = want[mode].copy() if mode else np.concatenate([want['rte'][:, :8], want['rtp'][:, 8:]], 1)
        if row == 17:
            w[:, 3::4] = 0
        d = got[:, row]
        wm = w & 0x7fff
        # NaN for NaN; section 10 item 7: a denormal half result may be 0
        ok = (d == w) | (np.isnan(x) & ((d & 0x7c00) == 0x7c00) & ((d & 0x3ff) != 0))
        ok |= (wm > 0) & (wm < 0x400) & ((d & 0x7fff) == 0)
        if row == 17:
            ok[:, 3::4] = d[:, 3::4] == 0
        rows = [((float(x[i, k]),), hex(int(d[i, k])), None if ok[i, k] else 'want %s' % hex(int(w[i, k])))
                for i in range(HL_N) for k in range(16)]
        errs += collect(name, rows)
    return errs


MACROS = {0: 110, 1: 100, 2: 110, 3: 1, 4: -1, 5: 1, 6: -1, 8: 1, 9: 8, 10: 127, 11: -128,
          12: 2 ** 31 - 1, 13: -2 ** 31, 14: 127, 15: -128, 16: 32767, 17: -32768, 18: 255,
          19: 65535, 20: -1, 21: 1, 22: 2, 23: 4, 24: 4, 25: 8, 26: 16, 27: 1, 28: 2, 29: 4,
          30: 4, 31: 16, 32: 4, 33: 32, 34: 4, 35: 4, 36: 4, 37: 4, 38: 4, 40: 2, 41: 1,
          # the extensions the device reports (runtime/cl/core.c), and not the others
          42: 0, 43: 0, 44: 0, 45: 1, 46: 1, 47: 1}
MACRO_NAMES = ('__OPENCL_VERSION__ CL_VERSION_1_0 CL_VERSION_1_1 __ENDIAN_LITTLE__ __IMAGE_SUPPORT__ '
               '__EMBEDDED_PROFILE__ __FAST_RELAXED_MATH__ __LINE__ __FILE__ CHAR_BIT CHAR_MAX '
               'CHAR_MIN INT_MAX INT_MIN SCHAR_MAX SCHAR_MIN SHRT_MAX SHRT_MIN UCHAR_MAX USHRT_MAX '
               'UINT_MAX vec_step(float) vec_step(int2) vec_step(char3) vec_step(uchar4) '
               'vec_step(short8) vec_step(float16) sizeof(char) sizeof(short) sizeof(int) '
               'sizeof(float) sizeof(float3) sizeof(uchar3) sizeof(short16) sizeof(size_t) '
               'sizeof(ptrdiff_t) sizeof(intptr_t) sizeof(uintptr_t) sizeof(int*) sizeof(bool) '
               'sizeof(half) sizeof(event_t) cl_khr_fp16 cl_khr_fp64 cles_khr_int64 '
               'cl_khr_global_int32_base_atomics cl_khr_local_int32_base_atomics '
               'cl_khr_byte_addressable_store').split()


def macros_check(ins, ref, dev):
    o = dev[0].reshape(-1)
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kernels', 'ep_misc.cl')).read()
    want = dict(MACROS)
    want[7] = src[:src.index('o[7] = __LINE__')].count('\n') + 1
    errs = ['%s = %d, want %d' % (MACRO_NAMES[k], o[k], v) for k, v in sorted(want.items()) if o[k] != v]
    print('    sizeof(bool) = %d' % o[39])
    return errs


def misc_cases():
    rng = np.random.default_rng(6117)
    h = half_inputs()
    cases = [
        Case('ep_h_load', K + 'ep_misc.cl', 'h_load', (HL_N,),
             [Buf('u2', HL_N * 16, values=h), Buf('u2', 32 * 16, values=h[:512]),
              Buf('f4', HL_N * 15 * 16, 'zeros'), Local(32)], check=half_load_check, skip_compare=(2,)),
        Case('ep_h_store', K + 'ep_misc.cl', 'h_store', (HL_N,),
             [Buf('f4', HL_N * 16, values=half_store_inputs()), Buf('u2', HL_N * 23 * 16, 'zeros'),
              Local(32)], check=half_store_check, skip_compare=(1,)),
    ]
    for t in ('char', 'uchar', 'short', 'ushort', 'int', 'uint'):
        dt = CV_TYPES[t][0]
        n = 64 // np.dtype(dt).itemsize
        cases.append(Case('ep_vls_' + t, K + 'ep_misc.cl', 'v_' + t, (256,),
                          [Buf(dt, 256 * n), Buf(dt, 8 * n), Buf(dt, 5 * n * 256, 'zeros'), Local(64)]))
    pow2 = np.ldexp(rng.choice([-1.0, 1.0], 128), rng.integers(-4, 5, 128))
    cases.append(Case('ep_vls_float', K + 'ep_misc.cl', 'v_float', (256,),
                      [Buf('f4', 256 * 16), Buf('f4', 128, values=pow2), Buf('f4', 6 * 16 * 256, 'zeros'),
                       Local(64)]))
    cases.append(Case('ep_async_copy', K + 'ep_misc.cl', 'a_copy', (64,),
                      [Buf('i4', 64 * 64), Buf('i4', 4 * 64 * 64, 'zeros'), Local(256)]))
    all_mn = [(m, n) for m in (2, 4, 8, 16) for n in (2, 4, 8, 16)]
    for t, u, mns in (('char', 'uchar', all_mn), ('int', 'uint', all_mn), ('float', 'uint', all_mn),
                      ('uchar', 'uchar', [(4, 4), (16, 16)]), ('short', 'ushort', [(4, 4), (16, 16)]),
                      ('ushort', 'ushort', [(4, 4), (16, 16)]), ('uint', 'uint', [(4, 4), (16, 16)])):
        dt, du = CV_TYPES[t][0], CV_TYPES[u][0]
        for m, n in mns:
            name = '%s_%d_%d' % (t, m, n)
            cases.append(Case('ep_shuffle_' + name, K + 'ep_shuffle.cl', 's_' + name, (256,),
                              [Buf((dt, 16), 256), Buf((dt, 16), 256), Buf((du, 16), 256),
                               Buf(dt, 32 * 256, 'zeros')]))
    cases += [
        Case('ep_macros', K + 'ep_misc.cl', 'm_macros', (1,), [Buf('i4', 48, 'zeros')],
             check=macros_check, skip_compare=(0,)),
        Case('ep_kernel_exec', K + 'ep_misc.cl', 'm_exec', (4,), [Buf('i4', 4, 'zeros')]),
        Case('ep_reqd_wg_size', K + 'ep_misc.cl', 'm_reqd', (4,), [Buf('i4', 4, 'zeros')]),
    ]
    return cases


EP_CASES += misc_cases()
